ratios = [
    'Halkomelem',
    'Zapotec',
    'Malagasy',
    'Maori',
    'Zinacantec Tzotzil',
    'Irish',
    'Bahasa',
    'Vietnamese',
    'Hebrew',
    'Swahili',
    'Romanian',
    'Greek',
    'Spanish'
    'Polish',
    'Korean',
    'Japanese',
    'Tamil',
    'Chinese',
    'English', 
    'Czech',
    'German',
    'Basque',
    'Archi',
    'Telugu',
    'Latin',
    'Dutch',
    'Tsez',
    'Hungarian'
]

heads = [
    'Japanese',
    'Korean',
    'Tamil',
    'German',
    'Persian',
    'Latin',
    'Tsez',
    'Baseque',
    'Germanic',
    'Tongan',
    'Mayan',
    'Irish',
    'Indonesian',
    'Yucatec',
    'English',
    'Russian',
    'Romance',
    'Bantoid'
]

def islang(df,langs):
    return df[(df['family'].isin(langs)) | (df['genus'].isin(langs)) | (df['Name'].isin(langs))].index
