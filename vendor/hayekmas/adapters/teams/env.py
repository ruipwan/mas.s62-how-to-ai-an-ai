from dataclasses import dataclass
import json
import math
import random

from hayekmas.base.env import BaseEnv


@dataclass(frozen=True)
class Task:
    id: str
    problem: str
    answer: str | int | float
    subject: str = ""

    def public(self):
        return {"id": self.id, "problem": self.problem, "subject": self.subject}


class ExactTaskEnv(BaseEnv):
    """Several auctioned actions per task; reference answers remain private."""

    def __init__(self, task, reward):
        super().__init__()
        self.task = task
        self.reward = reward
        self.score = 0.0

    def initialize(self):
        self.step_count = 0
        self.terminated = False
        self.action_history = {}
        self.score = 0.0

    def apply(self, action):
        if self.terminated:
            raise RuntimeError("Task already settled")
        self.step_count += 1
        self.action_history[self.step_count] = {
            "author": action.author,
            "text": str(action.answer),
            "final": action.final,
        }
        if not action.final:
            return 0.0
        answer = action.answer
        if isinstance(self.task.answer, (int, float)) and not isinstance(self.task.answer, bool):
            try:
                number = float(answer) if not isinstance(answer, bool) else float("nan")
                correct = math.isfinite(number) and math.isclose(number, self.task.answer, rel_tol=0, abs_tol=1e-8)
            except (TypeError, ValueError):
                correct = False
        else:
            correct = isinstance(answer, str) and answer.strip().casefold() == self.task.answer.strip().casefold()
        self.score = float(correct)
        self.terminated = True
        return self.reward * self.score

    def get_task_description(self):
        return self.task.problem

    def get_correct_answer(self):
        """Training repair reference; never included in task.public()."""
        return str(self.task.answer)

    def get_state_description(self):
        return json.dumps({"steps": self.action_history}, ensure_ascii=False)

    def get_terminal_score(self):
        return self.score


class ResearchTaskEnv(BaseEnv):
    """Use EoM's original ResearchEnv grader through a small action bridge.

    No specialized ResearchAgent is instantiated. The native action's `answer`
    role is an environment protocol tag, never an identity shown to agents.
    """

    def __init__(self, task, reward, judge):
        super().__init__()
        from hayekmas.adapters.researchworld.env import ResearchEnv
        from hayekmas.base.config import RewardConfig

        self.task = task
        self.native = ResearchEnv(
            task.problem,
            str(task.answer),
            reward_config=RewardConfig(terminal_output_bonus_scale=reward, center_env_reward=False),
        )
        self.judge_error = None

        def checked_judge(prompt):
            try:
                import re

                response = judge(prompt)
                match = re.search(r"(?m)^SCORE:\s*(\S+)", response)
                if not match or not math.isfinite(float(match[1])) or not 0 <= float(match[1]) <= 1:
                    raise ValueError("Judge must return a finite SCORE in [0, 1]")
                return response
            except Exception as error:
                self.judge_error = error
                raise

        self.native.llm_fn = checked_judge

    def initialize(self):
        self.native.initialize()
        self.judge_error = None

    def apply(self, action):
        from hayekmas.adapters.researchworld.agent import ResearchAction

        native_action = ResearchAction(str(action.answer), action.author, role="answer")
        native_action.is_final = action.final
        native_action.final_answer_text = str(action.answer) if action.final else None
        reward = self.native.apply(native_action)
        # Upstream's grader catches exceptions; surface these instead of counting a
        # transport failure or malformed judge output as a wrong solution.
        if self.judge_error is not None:
            raise self.judge_error
        return reward

    def get_task_description(self):
        return self.native.get_task_description()

    def get_correct_answer(self):
        """Forward the reference for training-only repair, not solver observations."""
        return self.native.get_correct_answer()

    def get_state_description(self):
        return self.native.state

    def get_terminal_score(self):
        return self.native.get_terminal_score() or 0.0

    def reach_termination(self):
        return self.native.reach_termination()


def load_tasks(path, split):
    tasks, seen = [], set()
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if "split" in row and row["split"] != split:
                continue
            identifier = row.get("id", row.get("task_group_id"))
            if identifier is None or not str(identifier).strip():
                raise ValueError("Task requires id or task_group_id")
            task = Task(
                str(identifier), row["problem"], row.get("answer", row.get("rubric")), str(row.get("subject", ""))
            )
            if task.id in seen or not isinstance(task.problem, str) or not task.problem.strip():
                raise ValueError("Task IDs must be unique and problems must be nonempty strings")
            if type(task.answer) not in (str, int, float) or (
                isinstance(task.answer, (int, float)) and not math.isfinite(task.answer)
            ):
                raise ValueError("Answers must be finite numbers or strings")
            seen.add(task.id)
            tasks.append(task)
    if not tasks:
        raise ValueError(f"No tasks for split {split!r}")
    return tasks


def demo_tasks(seed, rounds):
    rng = random.Random(seed)
    tasks = []
    for i in range(rounds):
        values = [rng.randint(-20, 40) for _ in range(5)]
        tasks.append(Task(f"sum-{seed}-{i}", f"Return the sum of these integers: {values}", sum(values)))
    return tasks
