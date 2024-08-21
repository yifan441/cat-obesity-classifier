import os

import pandas as pd

images_path = "dataset"
save_path_train = os.path.join(images_path, "annotations.csv")
test_path = os.path.join(images_path, 'test')
save_path_test = os.path.join(test_path, "annotations_test.csv")


def create_annotations_file(
    datadir: str, save_path: str
) -> None:
    print(f"creating annotations file from {datadir}...")
    class_labels = {"skinny": 0, "normal": 1, "obese": 2}
    annotations = {"image": [], "label": []}
    for img_class in os.listdir(datadir):
        if img_class not in class_labels:
            continue
        class_path = os.path.join(images_path, img_class)
        label = class_labels[img_class]
        for image in os.listdir(class_path):
            if image == ".DS_Store":
                continue
            image_path = os.path.join(class_path, image)
            annotations["image"].append(image_path)
            annotations["label"].append(label)

    pd.DataFrame(annotations).to_csv(save_path, index=False)
    print(f"Done! Annotations files saved at {save_path}.")


def main() -> None:
    create_annotations_file(images_path, save_path_train)
    create_annotations_file(test_path, save_path_test)


if __name__ == "__main__":
    main()
