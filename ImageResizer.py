import cv2


class ImageResizer:
    def resize_and_grayscale(image, destination_dir, resolution):
        img = cv2.imread(image)
        resized_image = cv2.resize(img, resolution, interpolation=cv2.INTER_LINEAR)
        resized_grayscale = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        destination_path = f"{destination_dir}/{image.split('/')[-1]}"
        cv2.imwrite(destination_path, resized_grayscale)

    def resize_grayscale_and_flip(image, destination_dir, resolution):
        img = cv2.imread(image)
        resized_image = cv2.resize(img, resolution, interpolation=cv2.INTER_LINEAR)
        resized_grayscale = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        flipped_image = cv2.flip(resized_grayscale, 1)

        destination_path = f"{destination_dir}/{image.split('/')[-1]}"
        cv2.imwrite(destination_path, flipped_image)
