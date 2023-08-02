from pdf2image import convert_from_path

def convert_pdf_to_images(pdf_path, output_folder):
    # Convert the PDF to images
    images = convert_from_path(pdf_path)

    for i, image in enumerate(images):
        # Save the image to the output folder
        image.save(f'{output_folder}/output_{i}.png', 'PNG')

pdf_path = "C:/Users/Rojus/Desktop/Test_creator/text.pdf"
image_folder = "C:/Users/Rojus/Desktop/Test_creator/pdf_images"

convert_pdf_to_images(pdf_path, image_folder)
