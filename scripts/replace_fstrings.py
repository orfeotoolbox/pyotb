#!/usr/bin/env python3
import re
import sys

assert sys.version_info.minor >= 6


def replace_fstrings(py_file, out_file):
    """
    Replace fstrings with .format() method
    :param py_file: path for input python file to convert
    :param out_file: converted file output path
    """

    def replace(line):
        exp = re.compile(r"""(.*[\s\(\[{=])f(['"])(.*)(['"])(.*)\n""")
        res = re.match(exp, line)
        if not res:
            return line
        prefix, qt1, string, qt2, suffix = res.groups()
        args = []
        for part in string.split('{')[1:]:
            a = part.split('}')[0]
            string = string.replace(a, '')
            args.append(a)

        return f"""{prefix}{qt1}{string}{qt2}.format({', '.join(args)}){suffix}\n"""

    with open(py_file, 'r') as r:
        lines = list(map(replace, r.readlines()))
    with open(out_file, 'w') as w:
        for li in lines:
            w.write(li)

if __name__ == "__main__":
    replace_fstrings(sys.argv[1], sys.argv[2])
