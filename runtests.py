import nose


argv =  ['', '-v',
         '--with-coverage', '--cover-package=simpeml', '--cover-erase',
         '--cover-tests', '--cover-branches']
nose.main(argv=argv)
