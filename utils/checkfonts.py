import matplotlib.font_manager
font_list = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
nanum_list = [matplotlib.font_manager.FontProperties(fname=font).get_name() for font in font_list if 'Nanum' in font]

if __name__ == "__main__":
    print(nanum_list)
