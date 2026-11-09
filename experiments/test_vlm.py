from src.reasoning.vlm import VisionLanguageModel


def main():

    vlm = VisionLanguageModel()

    image_path = (
        "data/processed/sample/frame_000007.jpg"
    )

    question = (
        "Describe what the person is doing in this image."
    )

    answer = vlm.answer_image(
        image_path=image_path,
        question=question,
    )

    print("\nQuestion:", question)
    print("Answer:", answer)


if __name__ == "__main__":
    main()