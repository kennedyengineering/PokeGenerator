import pathlib

def arborescence_to_dict(root):
  """Takes a pathlib.Path object and returns a dictionary representing the arborescence.
  Each node has key 'stop' which is associated with the Path to that node in particular."""
  subdirectories = [x for x in root.iterdir() if x.is_dir()]
  if len(subdirectories) == 0:
    return root
  result = {p.parts[-1]:arborescence_to_dict(p) for p in subdirectories}
  result['stop'] = root
  return result



def get_directories(data_dir):
    pokedir = arborescence_to_dict(data_dir)
    return [

        pokedir['pokemon']['versions']['generation-vii']['ultra-sun-ultra-moon']['stop'],               #this block is all rgba
        pokedir['pokemon']['versions']['generation-vii']['ultra-sun-ultra-moon']['shiny']['stop'],
        pokedir['pokemon']['versions']['generation-vii']['ultra-sun-ultra-moon']['shiny']['female'],
        pokedir['pokemon']['versions']['generation-vii']['ultra-sun-ultra-moon']['female'],

        pokedir['pokemon']['versions']['generation-vi']['x-y']['stop'],                                   #nearly all rgb, a few rgba
        pokedir['pokemon']['versions']['generation-vi']['x-y']['shiny']['stop'],
        pokedir['pokemon']['versions']['generation-vi']['x-y']['shiny']['female'],
        pokedir['pokemon']['versions']['generation-vi']['x-y']['female'],

        pokedir['pokemon']['versions']['generation-vi']['omegaruby-alphasapphire']['stop'],                 #more rgba than rgb, but split
        pokedir['pokemon']['versions']['generation-vi']['omegaruby-alphasapphire']['shiny']['stop'],
        pokedir['pokemon']['versions']['generation-vi']['omegaruby-alphasapphire']['shiny']['female'],
        pokedir['pokemon']['versions']['generation-vi']['omegaruby-alphasapphire']['female'],

        pokedir['pokemon']['versions']['generation-v']['black-white']['stop'],                 #this one throws errors sometimes
        pokedir['pokemon']['versions']['generation-v']['black-white']['shiny']['stop'],           #mostly rgb
        pokedir['pokemon']['versions']['generation-v']['black-white']['shiny']['female'],
        pokedir['pokemon']['versions']['generation-v']['black-white']['female'],

        pokedir['pokemon']['versions']['generation-iv']['heartgold-soulsilver']['stop'],          #all rgba
        pokedir['pokemon']['versions']['generation-iv']['heartgold-soulsilver']['shiny']['stop'],
        pokedir['pokemon']['versions']['generation-iv']['heartgold-soulsilver']['shiny']['female'],
        pokedir['pokemon']['versions']['generation-iv']['heartgold-soulsilver']['female'],

        pokedir['pokemon']['versions']['generation-iv']['diamond-pearl']['stop'],               #all rgba
        pokedir['pokemon']['versions']['generation-iv']['diamond-pearl']['shiny']['stop'],
        pokedir['pokemon']['versions']['generation-iv']['diamond-pearl']['shiny']['female'],
        pokedir['pokemon']['versions']['generation-iv']['diamond-pearl']['female'],

        pokedir['pokemon']['versions']['generation-iv']['platinum']['stop'],                  #all rgb
        pokedir['pokemon']['versions']['generation-iv']['platinum']['shiny']['stop'],
        pokedir['pokemon']['versions']['generation-iv']['platinum']['shiny']['female'],
        pokedir['pokemon']['versions']['generation-iv']['platinum']['female'],

    ]



def get_blacklist(data_dir):
   blacklist =  [
      '/pokemon/versions/generation-v/black-white/10186.png'
   ]
   return [pathlib.PosixPath(data_dir + s) for s in blacklist]