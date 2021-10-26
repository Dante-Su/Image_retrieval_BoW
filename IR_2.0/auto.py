import os, sys
  
def file_name(file_dir):   
    for root, dirs, files in os.walk(file_dir):  

        return files

name = file_name('dataset/testing/')

# len(name)

for i in range(len(name)):
    str=('python search_new.py -i dataset/testing/' + name[i])
    p=os.system(str)
    if i >= 3:
        print('The retrieval of {0}th image has been finished.'.format(i+1))
    elif i == 2:
        print('The retrieval of 3rd image has been finished.')
    elif i == 1:
        print('The retrieval of 2nd image has been finished.')
    elif i == 0:
        print('The retrieval of 1st image has been finished.')
    else:
        print('Error')
        sys.exit(0)