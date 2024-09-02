from config import config
from evaluation import load_resnet18, predict_img
from torchvision.io import read_image

MODEL_PATH = config.MODEL_SAVE_PATH
model = load_resnet18(MODEL_PATH)
model.cpu()

KEY = config.KEY


def main() -> None:
    print("Welcome to LoafTest!")
    img_path = input("Please enter the path to your cat image: ")
    print(
        "Analyzing... This might take a moment. Is your cat always this photogenic, or just today?"
    )

    image = read_image(img_path)
    pred = predict_img(model, image)

    match pred:
        case 0:
            output = "Oh no! Your cat is Underweight! Looks like someone might need a little extra kibble love."
        case 1:
            output = "Your cat is... drumroll please... Healthy! Just the right mix of fluff and finesse!"
        case 2:
            output = "HO-LY! That's a total LOAF! It’s possible someone’s been sneaking a few too many extra treats!"
        case _:
            raise RuntimeError("Unexpected Class!")

    print(output)


if __name__ == "__main__":
    main()
