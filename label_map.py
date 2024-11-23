label_map = {
    'BAS': 0, 'EBO': 1, 'EOS': 2, 'KSC': 3, 'LYA': 4, 'LYT': 5,
    'MMZ': 6, 'MOB': 7, 'MON': 8, 'MYB': 9, 'MYO': 10, 'NGB': 11,
    'NGS': 12, 'PMB': 13, 'PMO': 14
}

class_dict = {'NGS':'Neutrophil (segmented)',
      'NGB':'Neutrophil (band)',
      'EOS':'Eosinophil',
      'BAS':'Basophil',
      'MON':'Monocyte',
      'LYT':'Lymphocyte (typical)',
      'LYA':'Lymphocyte (atypical)',
      'KSC':'Smudge Cell',
      'MYO':'Myeloblast',
      'PMO':'Promyelocyte',
      'MYB':'Myelocyte',
      'MMZ':'Metamyelocyte',
      'MOB':'Monoblast',
      'EBO':'Erythroblast',
      'PMB':'Promyelocyte (bilobed)'}

class_names = [class_dict[key] for key in sorted(label_map.keys(), key=lambda x: label_map[x])]




