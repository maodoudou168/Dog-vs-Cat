import os
'''重命名train文件夹'''
root = './data/train'

a = 0
for item in os.listdir(root):
    if 'dog' in item:
        item_1 = 'dog.' + str(a) + '.jpg'
        # os.rename(os.path.join(root, item), os.path.join(root, item_1))
        dst = os.path.join(root, item_1)
        src = os.path.join(root, item)
        os.rename(src, dst)
    a += 1

'''重命名val文件夹'''
root = './data/eva'

i = 0
for item in os.listdir(root):
    if 'dog' in item:
        item_1 = 'dog.' + str(i) + '.jpg'
    else:
        item_1 = 'cat.' + str(i) + '.jpg'
    dst = os.path.join(root, item_1)
    src = os.path.join(root, item)
    os.rename(src, dst)
    i += 1

