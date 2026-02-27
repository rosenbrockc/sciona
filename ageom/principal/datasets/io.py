"""YAML adapter/template loader with recursive file linking and variable substitution.

Extracted from ``ageom.datasets.io`` with remote studio references removed.
"""

import yaml
from os import path

from ageom.datasets.parser.utils import chdir


REF_CHARACTER = "^"
"""str: character that indicated a remote studio reference in the original module.
In this standalone version, encountering it raises ``NotImplementedError``.
"""
LINK_CHARACTER = ":"
"""str: character to indicate that we want to load a local file as a nested
specification.
"""


def is_link(obj):
    """Determines whether the specified object is a link according to the
    templating specification.
    """
    result = False
    if isinstance(obj, str):
        if len(obj) > 0:
            result = obj[0] in [LINK_CHARACTER, REF_CHARACTER]
    return result


def substitute_varset(o: str, varset: dict = None):
    """Substitutes references to any of the variables in `varset` with their
    given values in the string `o`.
    """
    if varset is None:
        return o

    for v, r in varset.items():
        if '$' not in o:
            break

        o = o.replace(f"$({v})", r)

    return o


def _unpack_obj(context, obj, lcontext=None, recursive=True, varset: dict = None):
    """Unpacks each item of the specified object recursively so that all
    dictionary values are visited and all list items are also visited.

    .. warning:: `obj` will be mutated if any value it considers turns out to be
      a link (according to :func:`is_link`). In that case, the file descriptor
      will be placed by the actual contents of the YAML file that the link
      points to.

    Args:
        context (str): path to the root folder where the yaml file is
          located. Needed for relative paths of file links.
        lcontext (dict): local context for the items in `obj`. Keys are the
          names of keys in `obj`; values are relative folder paths that should
          be used as the context for reads within that item.
        varset: if any template values should have variable values substituted, specify
            the key-value mappings here.
    """
    if isinstance(obj, dict):
        result = obj
        for k, o in obj.items():
            ncontext = context
            #If the template specifies a relative context for this item,
            #then switch out the context for all of its children.
            if lcontext is not None and k in lcontext:
                with chdir(context):
                    ncontext = path.abspath(lcontext[k])

            if varset is not None and isinstance(o, str) and '$' in o:
                o = substitute_varset(o, varset)

            if is_link(o):
                result[k] = read(ncontext, o, recursive=recursive)
            else:
                result[k] = _unpack_obj(ncontext, o, recursive=recursive, varset=varset)
    elif isinstance(obj, (list, set, tuple)):
        result = []
        for o in obj:
            if varset is not None and isinstance(o, str) and '$' in o:
                o = substitute_varset(o, varset)

            if is_link(o):
                result.append(read(context, o, recursive=recursive))
            else:
                result.append(_unpack_obj(context, o, recursive=recursive, varset=varset))
    elif varset is not None and isinstance(obj, str) and '$' in obj:
        result = substitute_varset(obj, varset)
    else:
        result = obj

    return result


def read(context, yfile, recursive=True, varset: dict = None):
    """Reads in the specified YAML file, following any additional file
    directives to compile a full representation of the template hierarchy for
    the root file.

    Args:
        context (str): path to the root folder where the yaml file is
            located. Needed for relative paths of file links.
        yfile (str): name of the template YAML file *relative* to
            `context`. Should *not* include the `.yaml` or `.yml` extension.
        varset: if any template values should have variable values substituted, specify
            the key-value mappings here.
    """
    target = None
    with chdir(context):
        if yfile[0] == LINK_CHARACTER:
            if recursive:
                root = path.abspath(yfile[1:])
            else:
                return {}
        elif yfile[0] == REF_CHARACTER:
            raise NotImplementedError(
                f"Remote studio references ('{REF_CHARACTER}') are not supported "
                f"in principal/datasets. Use a local '{LINK_CHARACTER}' link or a "
                f"direct file path instead. Got: {yfile!r}"
            )
        else:
            root = path.abspath(yfile)

    if target is None and path.isfile(root + ".yml"):
        target = root + ".yml"
    elif target is None:
        emsg = ("The specified template file '{}' was not found relative "
                "to the given context directory ('{}'). Note that all files"
                " should use the `.yml` extension, *not* `.yaml`.")
        raise ValueError(emsg.format(yfile, context))

    with open(target, 'r') as stream:
        result = yaml.load(stream, Loader=yaml.FullLoader)

    #Determine the new context for recursive file links within the values of
    #this file.
    ncontext = path.dirname(target)

    #The specification allows for a "local" context that describes folder
    #locations for specific items within the template.
    lcontext = None
    if isinstance(result, dict) and "context" in result:
        lcontext = result["context"]
        del result["context"]

    #The unpacking command will mutate the values in result so that file links
    #are expanded to be full-fledged python objects from their YAML files.
    result = _unpack_obj(ncontext, result, lcontext, recursive=recursive, varset=varset)
    return result
