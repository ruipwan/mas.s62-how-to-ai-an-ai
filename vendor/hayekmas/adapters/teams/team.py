"""Membership mechanics independent of bidding, solving and reflection."""

from dataclasses import dataclass, field


@dataclass
class Team:
    team_id: str
    member_ids: list[str]
    created_at: int
    summary: str = ""
    trajectory: list = field(default_factory=list)
    contributions: dict[str, float] = field(default_factory=dict)

    def total_wealth(self, population):
        return sum(a.wealth for a in population if a.name in self.member_ids)


class TeamManager:
    def __init__(self, population, max_size):
        self.population = population
        self.max_size = max_size
        self.teams = {}
        self.invitations = {}
        self.team_counter = 0
        self.invitation_counter = 0

    def agent(self, name):
        return next((a for a in self.population if a.name == name), None)

    def active_teams(self, episode):
        grouped = {}
        for agent in self.population:
            key = agent.team_tag or f"solo:{agent.name}"
            grouped.setdefault(key, []).append(agent.name)
        for key, names in grouped.items():
            if key not in self.teams:
                self.teams[key] = Team(key, names, episode)
            self.teams[key].member_ids = names
        return [self.teams[key] for key in grouped]

    def create_team(self, agents, episode):
        if len(agents) > self.max_size or len(agents) < 2 or any(a.team_tag for a in agents):
            raise ValueError("Invalid team creation")
        self.team_counter += 1
        tag = f"team-{self.team_counter}"
        for agent in agents:
            agent.team_tag = tag
        self.teams[tag] = Team(tag, [a.name for a in agents], episode)
        return tag

    def invite(self, inviter, invitee, episode):
        if invitee is inviter or invitee.team_tag is not None:
            raise ValueError("Recipient must be a different, unteamed agent")
        size = sum(a.team_tag == inviter.team_tag for a in self.population) if inviter.team_tag else 1
        if size >= self.max_size:
            raise ValueError("Engineering team-size cap reached")
        self.invitation_counter += 1
        invitation = {
            "id": f"invite-{self.invitation_counter}",
            "from": inviter.name,
            "to": invitee.name,
            "team": inviter.team_tag,
            "round": episode,
        }
        self.invitations[invitation["id"]] = invitation
        return invitation

    def accept_invite(self, invitee, invitation_id, episode):
        invitation = self.invitations.get(invitation_id) if isinstance(invitation_id, str) else None
        inviter = self.agent(invitation["from"]) if invitation else None
        if (
            not invitation
            or invitation["to"] != invitee.name
            or invitee.team_tag is not None
            or invitation["round"] != episode
            or inviter.team_tag != invitation["team"]
        ):
            raise ValueError("Invalid, stale, or unauthorized invitation")
        if inviter.team_tag is None:
            tag = self.create_team([inviter, invitee], episode)
        else:
            members = [a for a in self.population if a.team_tag == inviter.team_tag]
            if len(members) >= self.max_size:
                raise ValueError("Engineering team-size cap reached")
            tag = inviter.team_tag
            invitee.team_tag = tag
        del self.invitations[invitation_id]
        self.active_teams(episode)
        return tag, inviter.name

    def leave_team(self, agent):
        old = agent.team_tag
        agent.team_tag = None
        remaining = [a for a in self.population if old and a.team_tag == old]
        dissolved = None
        if len(remaining) == 1:
            remaining[0].team_tag = None
            dissolved = remaining[0].name
        if old in self.teams:
            self.teams[old].member_ids = [a.name for a in remaining] if len(remaining) > 1 else []
        return old, dissolved
