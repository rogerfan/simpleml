import sys, argparse

import nose


parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('-p', '--public', action='store_true')
parser.add_argument('-t', '--tests', action='store_true')

if 'runtests' in sys.argv[0]:
    args = parser.parse_args(sys.argv[1:])
else:
    args = parser.parse_args(sys.argv)

argv = ['', '--with-coverage', '--cover-erase',
        '--cover-tests', '--cover-html']
if args.verbose:
    argv.append('-v')
if args.tests or args.public:
    argv.append('--cover-package="simpleml,tests"')
    if args.public:
        argv.append('./tests/public_tests')
else:
    argv.append('--cover-package=simpleml')

nose.main(argv=argv)
