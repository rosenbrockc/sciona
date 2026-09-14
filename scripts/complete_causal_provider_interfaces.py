#!/usr/bin/env python3
"""Complete callable defaults and omitted optional ports for reviewed causal providers."""
import argparse
from collections import Counter
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
from uuid import uuid5
import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from scripts.import_residual_execution_drafts import ensure_row


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--apply',action='store_true');args=parser.parse_args();counts=Counter()
    with psycopg.connect(os.environ['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row) as db:
        records=db.execute("SELECT * FROM artifact_audit_evidence WHERE runner_version='causal-provider-intake.v1' ORDER BY evidence_id FOR SHARE").fetchall()
        for record in records:
            module,name=record['details']['runtime_fqdn'].rsplit('.',1);fn=getattr(importlib.import_module(module),name)
            if hashlib.sha256(Path(inspect.getfile(inspect.unwrap(fn))).read_bytes()).hexdigest()!=record['details']['source_sha256']:raise ValueError('provider source changed')
            signature=inspect.signature(fn)
            for table,key in [('artifact_io_specs','artifact_id'),('atom_io_specs','atom_id')]:
                for ordinal,(name,param) in enumerate(signature.parameters.items()):
                    required=param.default is inspect.Parameter.empty;default='' if required else repr(param.default)
                    row=db.execute(sql.SQL("SELECT * FROM {} WHERE version_id=%s AND direction='input' AND name=%s FOR UPDATE").format(sql.Identifier(table)),(record['version_id'],name)).fetchone()
                    if row:
                        if row['required']!=required or row['ordinal']!=ordinal:raise ValueError('existing input order or required flag differs')
                        if row['default_value_repr'] not in (None,'',default):raise ValueError('existing default differs')
                        if row['default_value_repr']!=default:
                            db.execute(sql.SQL("UPDATE {} SET default_value_repr=%s WHERE version_id=%s AND direction='input' AND name=%s").format(sql.Identifier(table)),(default,record['version_id'],name));counts['defaults_completed']+=1
                    else:
                        if required and table!='atom_io_specs':raise ValueError('required input absent from original intake')
                        values={key:record['artifact_id'],'version_id':record['version_id'],'direction':'input','name':name,'ordinal':ordinal,'type_desc':type(param.default).__name__ if not required else 'object','required':False,'default_value_repr':default,'constraints':'Optional callable argument; default verified against the exact source version.'}
                        if required:
                            canonical=db.execute("SELECT type_desc,constraints FROM artifact_io_specs WHERE version_id=%s AND direction='input' AND name=%s",(record['version_id'],name)).fetchone()
                            if not canonical:raise ValueError('required canonical input missing')
                            values.update(canonical,required=True,default_value_repr='')
                        ensure_row(db,table,{k:values[k] for k in [key,'version_id','direction','name']},values);counts['ports_added']+=1
            outputs=db.execute("SELECT name,ordinal,type_desc,constraints,required,default_value_repr FROM artifact_io_specs WHERE version_id=%s AND direction='output' ORDER BY ordinal",(record['version_id'],)).fetchall()
            for output in outputs:
                values={'atom_id':record['artifact_id'],'version_id':record['version_id'],'direction':'output',**output}
                counts['output_ports_added']+=ensure_row(db,'atom_io_specs',{k:values[k] for k in ['atom_id','version_id','direction','name']},values)
            audit_id=uuid5(record['version_id'],'causal-provider-interface.v1')
            ensure_row(db,'artifact_audit_evidence',{'evidence_id':audit_id},{'evidence_id':audit_id,'artifact_id':record['artifact_id'],'version_id':record['version_id'],'audit_type':'asset_integrity_check','passed':True,'status':'completed','source_kind':'automated','runner_version':'causal-provider-interface.v1','details':Jsonb({'signature':str(signature),'source_sha256':record['details']['source_sha256'],'scope':'Complete ordered callable parameters and exact defaults in both catalog interfaces.'})})
            counts['providers_verified']+=1
        if not args.apply:db.rollback()
    print(json.dumps({'applied':args.apply,'counts':dict(counts)},indent=2))


if __name__=='__main__':main()
