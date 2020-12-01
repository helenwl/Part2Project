from PIL import Image


background = Image.open("defectivelatticegeneration\\defectiveMoS2lattice_cellholes_9a.png")
overlay = Image.open("defectivelatticegeneration\\contrast_image_9.png")

background = background.convert("RGBA")
overlay = overlay.convert("RGBA")

new_img = Image.blend(background, overlay, 0.4)
new_img.save("new.png","PNG")