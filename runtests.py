import nose


argv =  ['', '--with-coverage', '--cover-package=simpeml', '--cover-erase',
         '--cover-tests', '--cover-branches']
nose.main(argv=argv)
