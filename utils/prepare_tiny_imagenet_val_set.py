import os


val_dir = "../data/tiny-imagenet-200/val/"

val_image_dir = os.path.join(val_dir, "images")
images = [d for d in os.listdir(val_image_dir)]
val_annotations_file = os.path.join(val_dir, "val_annotations.txt")
val_img_to_class = {}
set_of_classes = set()
with open(val_annotations_file, 'r') as fo:
    entry = fo.readlines()
    for data in entry:
        words = data.split("\t")
        val_img_to_class[words[0]] = words[1]
        set_of_classes.add(words[1])

len_dataset = len(list(val_img_to_class.keys()))
classes = sorted(list(set_of_classes))

for cls in classes:
    if not os.path.exists(val_dir + cls):
        os.makedirs(val_dir + cls)

for k, v in val_img_to_class.items():
    src_path = os.path.join(val_image_dir, k)
    target_path = os.path.join(os.path.join(val_dir, v), k)
    os.rename(src_path, target_path)
