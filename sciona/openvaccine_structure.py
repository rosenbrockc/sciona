"""Independent loop labels for balanced pseudoknot-free dot-bracket strings."""


def loop_labels(structure: str) -> str:
    if not isinstance(structure,str) or not structure or set(structure)-set('.()'):
        raise ValueError('Nonempty dot-parenthesis structure required')
    stack=[];children={None:[]};dots={None:[]};partners={}
    for pos,token in enumerate(structure):
        parent=stack[-1] if stack else None
        if token=='(':
            children[parent].append(pos)
            children[pos]=[];dots[pos]=[];stack.append(pos)
        elif token==')':
            if not stack:raise ValueError('Unmatched closing parenthesis')
            partners[stack.pop()]=pos
        else:dots[parent].append(pos)
    if stack:raise ValueError('Unmatched opening parenthesis')
    labels=['S']*len(structure)
    for parent,positions in dots.items():
        if parent is None:label='E'
        elif not children[parent]:label='H'
        elif len(children[parent])>1:label='M'
        else:
            child=children[parent][0]
            left=any(p<child for p in positions)
            right=any(p>partners[child] for p in positions)
            label='I' if left and right else 'B'
        for pos in positions:labels[pos]=label
    return ''.join(labels)
