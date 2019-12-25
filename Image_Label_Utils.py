import os, cv2
import shutil, tqdm
from PIL import Image

def image_bbox_resize_ops(config, RAWDATA):
    main_folder = os.path.dirname(RAWDATA)
    
    ### following three folders correspond to the contents of "OID" inside the main project directory
    train_folder = os.path.join(RAWDATA, "Dataset", "train")
    test_folder = os.path.join(RAWDATA, "Dataset", "test")
    valid_folder = os.path.join(RAWDATA, "Dataset", "validation")
    
    foldername = [folder for folder in os.listdir(train_folder) if "Person" and "Car" in folder.split("_")]
    ### you should only have one folder for this task, else change the search words
    ### in the list comprehension above
    assert len(foldername) == 1
    foldername = foldername[0]

    ### Create the folder structure
    store_train_folder = os.path.join(config.maindir, "Dataset", config.PROJECT, "TRAIN")
    store_test_folder = os.path.join(config.maindir, "Dataset", config.PROJECT, "TEST")
    os.makedirs(store_train_folder, exist_ok=True)
    os.makedirs(store_test_folder, exist_ok=True)

    ### Putting the training and validation images & labels in Dataset/TRAIN folder
    ### Putting the testing images & labels in Dataset/TEST folder
    for task in ["train", "test", "valid"]:
        if task in ("train", "valid"):
            store_image_folder = os.path.join(store_train_folder, "Images")
            store_label_folder = os.path.join(store_train_folder, "Labels")
            os.makedirs(store_image_folder, exist_ok=True)
            os.makedirs(store_label_folder, exist_ok=True)
        else:
            store_image_folder = os.path.join(store_test_folder, "Images")
            store_label_folder = os.path.join(store_test_folder, "Labels")
            os.makedirs(store_image_folder, exist_ok=True)
            os.makedirs(store_label_folder, exist_ok=True)

        orig_folder = os.path.join(eval(task + "_folder"), foldername)
        images = [f for f in os.listdir(orig_folder) if f.endswith(".jpg")]
        for i in tqdm.tqdm(range(len(images)), total=len(images), desc="Resizing {} images and boxes".format(task.upper())):
            filename = images[i].split(".")[0]
            img = Image.open(os.path.join(orig_folder, images[i]))
            ## Note that img.size returns width and height. PIL returns an RGB Image object
            resize_ratio_x, resize_ratio_y = config.IMAGE_W/img.size[0], config.IMAGE_H/img.size[1]
            ### resize the Image (image object) and save to disk
            img = img.resize((config.IMAGE_W, config.IMAGE_H), resample=Image.LANCZOS)
            img.save(os.path.join(store_image_folder, filename + ".jpg"), format="JPEG")

            with open(os.path.join(orig_folder, "Label", filename + ".txt"), 'r') as f:
                label = f.read().split("\n")[:-1]

            with open(os.path.join(store_label_folder, filename + ".txt"), 'w') as f_w:
                for line in label:
                    class_label = "_".join(word for word in line.split(" ") if word.isalpha())
                    numbers = [float(num) for num in line.split(" ") if not num.isalpha()]
                    ### resize the bounding box here
                    numbers[0], numbers[1] = numbers[0]*resize_ratio_x, numbers[1]*resize_ratio_y
                    numbers[2], numbers[3] = numbers[2]*resize_ratio_x, numbers[3]*resize_ratio_y
                    numbers = list(map(lambda x: str(int(round(x))), numbers))
                    resized_line = " ".join([class_label] + numbers)
                    ### write the resized box information to a new text file
                    f_w.write(resized_line)
                    f_w.write("\n")
            ## finish writing the resized box information file
            f_w.close()
