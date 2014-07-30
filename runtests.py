import nose


argv =  ['', '-v',
         '--with-coverage', '--cover-package=simpleml', '--cover-erase',
         '--cover-tests', '--cover-branches']
nose.main(argv=argv)
