from pathlib import Path
from PIL import Image
from tqdm import tqdm

def tiff_to_jpg(
    input_dir,
    output_dir,
    quality=90,
    subsampling=0,
    optimize=True
):
    """
    Conversion en masse TIFF -> JPEG

     input_dir: répertoire contenant les TIFF
     output_dir: répertoire de sortie des JPEG
     quality: qualité JPEG 
     subsampling: 0 = pas de sous-échantillonnage 
     optimize: optimisation Huffman (nom de l'algorithme de compression)
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tiff_files = list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))

    for tiff_path in tqdm(tiff_files, desc="Conversion TIFF en JPEG"):
        try:
            with Image.open(tiff_path) as img:
                img = img.convert("RGB")

                jpg_path = output_dir / (tiff_path.stem + ".jpg")

                img.save(
                    jpg_path,
                    format="JPEG",
                    quality=quality,
                    subsampling=subsampling,
                    optimize=optimize,
                    progressive=False
                )

        except Exception as e:
            print(f"Erreur sur {tiff_path.name} : {e}")
tiff_to_jpg(
    "Datas/in/00_source_images/herbarium/1",
    "Datas/in/00_source_images/herbarium/jpeg",
    quality=90,
    subsampling=0
)