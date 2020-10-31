from configparser import ConfigParser

# Get the configparser object
config_object = ConfigParser()

# Add info config
config_object["DATA-INPUT"] = {
    "dataset": [
        "iris",
        "nuclear"
        "Landsat7neighbour",
        "landsatImg",
        "sat.all",
        "ulc",
    ]
}

config_object["GA-CONFIG"] = {
    "populationSIZE": 50,
    "generationNUM": 50,
    "matePB": 0.9,
    "mutatePB": 0.05
}

# Write the above sections to config.ini file
with open('config.ini', 'w') as conf:
    config_object.write(conf)
