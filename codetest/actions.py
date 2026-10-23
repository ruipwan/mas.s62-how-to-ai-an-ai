"""Strict provider output contracts and local validation of their small schema subset."""


def object_schema(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}


def action_schema(phase, shared_scratchpad=False, contribution_contracts=False):
    text = {'type': 'string'}
    strings = {'type': 'array', 'items': text}
    if phase == 'peer':
        return object_schema({'message': text})
    if phase == 'vote':
        return object_schema({'choice': {'type': ['string', 'null']}, 'message': text})
    if phase == 'orchestrate':
        return object_schema({'message': text, 'assignments': {'type': 'array', 'items': object_schema({
            'agent': text, 'task': text, 'inputs': strings})}})
    independent = phase in {'independent', 'synthesis'}
    if independent:
        phase = 'work'
    text = {'type': 'string'}
    strings = {'type': 'array', 'items': text}
    communication = {'message': text, 'channel': {'type': 'string', 'enum': ['team', 'public', 'direct']},
                     'recipients': strings}
    if phase == 'membership':
        fields = {'leave': {'type': 'boolean'}, 'invite': {'type': ['string', 'null']}, 'message': text}
    elif phase == 'join':
        fields = {'invitation': {'type': ['string', 'null']}, 'message': text}
    elif phase == 'discuss':
        fields = communication
    elif phase == 'ballot':
        fields = {'yes': {'type': 'boolean'}, 'pledge': {'type': 'number', 'enum': [0, .025, .05, .1]},
                  'authorize_fixed': {'type': 'boolean'}, 'message': text}
    elif phase == 'work':
        fields = {'tool': {'type': 'string', 'enum': ['inspect', 'edit', 'test', 'probe', 'message']},
                  'command': {'type': ['string', 'null']}, **communication,
                  'args': strings, 'stdin': text, 'capture': strings,
                  'files': {'type': 'array', 'items': object_schema({'path': text, 'content': text})}}
    else:
        raise ValueError('Unknown action phase')
    if independent:
        del fields['channel']
        del fields['recipients']
    if shared_scratchpad:
        note = object_schema({'key': text, 'revision': {'type': 'number'},
                              'kind': {'type': 'string', 'enum': ['task', 'finding', 'interface', 'blocker']},
                              'status': {'type': 'string', 'enum': ['open', 'claimed', 'blocked', 'done']},
                              'text': text, 'dependencies': strings})
        note['type'] = ['object', 'null']
        fields['scratchpad'] = note
        if phase == 'work':
            fields['tool']['enum'].append('fetch')
    if contribution_contracts and phase == 'work':
        contract = object_schema({
            'operation': {'type': 'string', 'enum': ['offer', 'request', 'fund', 'accept', 'deliver', 'settle', 'cancel']},
            'id': {'type': ['string', 'null']}, 'artifact': {'type': ['string', 'null']},
            'provider': {'type': ['string', 'null']}, 'description': text,
            'price': {'type': 'number', 'enum': [0, .025, .05, .1]},
            'paths': strings, 'probe_ids': strings,
            'expires_in': {'type': 'number', 'enum': [0, 1, 2, 3]}})
        contract['type'] = ['object', 'null']
        fields['contract'] = contract
        fields['tool']['enum'].append('contract')
    return object_schema(fields)


def validate(value, schema):
    types = schema['type'] if isinstance(schema['type'], list) else [schema['type']]
    names = {str: 'string', bool: 'boolean', int: 'number', float: 'number', dict: 'object', list: 'array', type(None): 'null'}
    if names.get(type(value)) not in types:
        raise ValueError('Action field has the wrong type')
    if 'enum' in schema and value not in schema['enum']:
        raise ValueError('Action field is outside the allowed choices')
    if isinstance(value, dict):
        if set(value) != set(schema['properties']):
            raise ValueError('Action fields do not match the required schema')
        for key, item in value.items():
            validate(item, schema['properties'][key])
    if isinstance(value, list):
        for item in value:
            validate(item, schema['items'])


def normalize_action(value, phase, shared_scratchpad=False, contribution_contracts=False):
    validate(value, action_schema(phase, shared_scratchpad, contribution_contracts))
    if phase in {'work', 'independent', 'synthesis'}:
        files = value['files']
        if len({f['path'] for f in files}) != len(files):
            raise ValueError('Duplicate probe file paths')
        value = {**value, 'files': {f['path']: f['content'] for f in files}}
    return value
