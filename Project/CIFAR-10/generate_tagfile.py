import os

# os.system('find cifar-10/train/ -type f > cifar-10/cifar_train_60000.txt')

# for root, dirname, filename in os.walk('cifar-10/train'):
#     train_folder = dirname
#     break

root_folder = 'cifar-10'

for train_test in ('train', 'test'):

    print("Generating", train_test, "set tagfile...")

    ## Method 1.
    # Take only one element from generator
    folder = sorted(next(os.walk(os.path.join(root_folder, train_test)))[1])

    ## Method 2.
    #folder = sorted(os.listdir(os.path.join(root_folder, train_test)))
    #try:
    #    # For macOS
    #    folder.remove('.DS_Store')
    #except:
    #    print('Not found .DS_Store')

    train_pic_path_tag = []

    for class_encode, classname in enumerate(folder):
        for root, dirname, filename in os.walk(os.path.join(root_folder, train_test, classname)):
            for pic in filename:
                train_pic_path_tag.append((str(classname) + '/' + pic, class_encode))

    with open(os.path.join(root_folder, 'cifar_' + train_test + '.txt'), 'w') as tagfile:
        for pic_path, pic_tag in train_pic_path_tag:
            tagfile.write(pic_path + ' ' + str(pic_tag) + '\n')

print("Generate class encode file...")

with open(os.path.join(root_folder, 'class_encode.txt'), 'w') as encodefile:
    for class_encode, classname in enumerate(folder):
        encodefile.write(classname + ' ' + str(class_encode) + '\n')
    
