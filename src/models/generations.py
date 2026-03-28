"""Car generation lookup: DBpedia SPARQL + static fallback for full coverage."""

import json
import logging
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_CACHE_PATH = _DATA_DIR / "generations.json"
_generations: dict | None = None
_CACHE_MAX_AGE = 24 * 3600  # refresh daily

# =========================================================================
# Brand normalization & model aliases
# =========================================================================

# DBpedia manufacturer label → standard brand name
_BRAND_NORMALIZE: dict[str, str] = {
    "Volkswagen Group": "Volkswagen", "FAW-Volkswagen": "Volkswagen",
    "Mercedes-Benz Group": "Mercedes-Benz", "Daimler-Benz": "Mercedes-Benz",
    "Audi AG": "Audi",
    "Ford Motor Company": "Ford", "Ford of Europe": "Ford",
    "Hyundai Motor Company": "Hyundai",
    "Jaguar Cars": "Jaguar", "Jaguar Land Rover": "Jaguar",
    "Lotus Cars": "Lotus", "Mitsubishi Motors": "Mitsubishi",
    "Renault Sport": "Renault", "Peugeot Sport": "Peugeot",
    "Dongfeng Peugeot-Citroën": "Citroën",
    "Nissan Motor Company": "Nissan",
    "Honda Motor Company": "Honda", "Toyota Motor Corporation": "Toyota",
    "Volvo Cars": "Volvo",
    "Tesla, Inc.": "Tesla",
    "Škoda Auto": "Škoda",
    "DS Automobiles": "DS",
    "Pontiac (automobile)": "Pontiac",
    "British Motor Corporation": "Mini",
    "SAIC Motor": "MG",
    "MG Cars": "MG", "MG Motor": "MG", "MG cars": "MG",
    "Stellantis Europe": "Fiat",
}

# Listing brand → canonical brand for lookup
_BRAND_LOOKUP: dict[str, str] = {
    "VW": "Volkswagen",
    "Citroën": "Citroen",
    "SEAT": "Seat",
}

_MODEL_ALIASES: dict[str, dict[str, str]] = {
    "BMW": {
        "116": "1 Series", "118": "1 Series", "120": "1 Series",
        "125": "1 Series", "130": "1 Series", "135": "1 Series",
        "218": "2 Series", "220": "2 Series", "225": "2 Series",
        "230": "2 Series",
        "316": "3 Series", "318": "3 Series", "320": "3 Series",
        "325": "3 Series", "328": "3 Series", "330": "3 Series",
        "335": "3 Series", "340": "3 Series",
        "418 Gran Coupé": "4 Series Gran Coupé",
        "420": "4 Series", "420 Gran Coupé": "4 Series Gran Coupé",
        "425": "4 Series", "430": "4 Series", "435": "4 Series",
        "440": "4 Series",
        "520": "5 Series", "525": "5 Series", "530": "5 Series",
        "535": "5 Series", "540": "5 Series", "550": "5 Series",
        "630 Gran Turismo": "6 Series Gran Turismo",
        "640": "6 Series", "650": "6 Series",
        "730": "7 Series", "740": "7 Series", "750": "7 Series",
    },
    "Mercedes-Benz": {
        "180": "C-Class", "220": "E-Class",
        "A 160": "A-Class", "A 180": "A-Class", "A 200": "A-Class",
        "A 220": "A-Class", "A 250": "A-Class",
        "B 180": "B-Class", "B 200": "B-Class", "B 220": "B-Class",
        "C 180": "C-Class", "C 200": "C-Class", "C 220": "C-Class",
        "C 250": "C-Class", "C 300": "C-Class", "C 350": "C-Class",
        "E 200": "E-Class", "E 220": "E-Class", "E 250": "E-Class",
        "E 300": "E-Class", "E 350": "E-Class",
        "CLA 180": "CLA-Class", "CLA 200": "CLA-Class",
        "CLA 220": "CLA-Class", "CLA 250": "CLA-Class",
        "CLA 45 AMG": "CLA-Class", "CLC 220": "CLC-Class",
        "GLA 180": "GLA-Class", "GLA 200": "GLA-Class",
        "GLA 220": "GLA-Class",
        "GLB 180": "GLB-Class", "GLB 200": "GLB-Class",
        "GLC 220": "GLC-Class", "GLC 250": "GLC-Class",
        "GLC 300": "GLC-Class",
    },
    "Audi": {
        "A1 Sportback": "A1", "A3 Sportback": "A3",
        "A4 Avant": "A4", "A5 Cabrio": "A5",
        "S3 Limousine": "S3", "Q4 Sportback e-tron": "Q4 e-tron",
    },
    "Opel": {"Astra Sports Tourer": "Astra"},
    "Renault": {
        "Clio Sport Tourer": "Clio", "Mégane Break": "Mégane",
        "Mégane E-Tech": "Mégane",
        "Grand Scénic": "Scénic", "Grand Modus": "Modus",
    },
    "Peugeot": {"e-208": "208", "307 SW": "307", "407 SW": "407", "508 SW": "508"},
    "Volkswagen": {"Golf Variant": "Golf", "ID.7": "ID.7"},
    "VW": {"Golf": "Golf"},
    "Seat": {"Ibiza ST": "Ibiza"},
    "Citroen": {
        "C4 Spacetourer": "C4", "C5 Break": "C5", "DS4": "DS4",
        "e-Mehari": "e-Mehari",
    },
    "Porsche": {"Panamera Sport Turismo": "Panamera", "718 Boxster": "718 Boxster and Cayman"},
    "Volvo": {"XC 90": "XC90"},
    "Land Rover": {"Evoque": "Range Rover Evoque"},
    "Smart": {"ForTwo Coupé": "ForTwo"},
    "Toyota": {"107": "Aygo"},
}

# =========================================================================
# Static generation data — fallback for models DBpedia doesn't cover
# =========================================================================

_STATIC_GENERATIONS: dict[str, dict[str, list]] = {
    "Seat": {
        "Ibiza": [
            {"name": "Mk2", "year_from": 1993, "year_to": 2002},
            {"name": "6L", "year_from": 2002, "year_to": 2008},
            {"name": "6J", "year_from": 2008, "year_to": 2017},
            {"name": "KJ1", "year_from": 2017, "year_to": 2026},
        ],
        "Leon": [
            {"name": "Mk1", "year_from": 1999, "year_to": 2005},
            {"name": "Mk2", "year_from": 2005, "year_to": 2012},
            {"name": "Mk3", "year_from": 2012, "year_to": 2020},
            {"name": "Mk4", "year_from": 2020, "year_to": 2026},
        ],
        "Toledo": [
            {"name": "1L", "year_from": 1991, "year_to": 1999},
            {"name": "1M", "year_from": 1999, "year_to": 2004},
            {"name": "5P", "year_from": 2004, "year_to": 2009},
            {"name": "KG", "year_from": 2012, "year_to": 2019},
        ],
    },
    "Citroen": {
        "C3": [
            {"name": "Mk1", "year_from": 2002, "year_to": 2009},
            {"name": "Mk2", "year_from": 2009, "year_to": 2016},
            {"name": "Mk3", "year_from": 2016, "year_to": 2024},
        ],
        "C4": [
            {"name": "Mk1", "year_from": 2004, "year_to": 2010},
            {"name": "Mk2", "year_from": 2010, "year_to": 2018},
            {"name": "Mk3", "year_from": 2020, "year_to": 2026},
        ],
        "C5": [
            {"name": "Mk1", "year_from": 2001, "year_to": 2007},
            {"name": "Mk2", "year_from": 2008, "year_to": 2017},
        ],
        "DS4": [
            {"name": "Gen1", "year_from": 2011, "year_to": 2018},
        ],
    },
    "Volvo": {
        "V40": [
            {"name": "Mk1", "year_from": 1995, "year_to": 2004},
            {"name": "Mk2", "year_from": 2012, "year_to": 2019},
        ],
        "V60": [
            {"name": "Mk1", "year_from": 2010, "year_to": 2018},
            {"name": "Mk2", "year_from": 2018, "year_to": 2026},
        ],
        "V90": [
            {"name": "Mk2", "year_from": 2016, "year_to": 2026},
        ],
        "XC90": [
            {"name": "Mk1", "year_from": 2002, "year_to": 2014},
            {"name": "Mk2", "year_from": 2014, "year_to": 2026},
        ],
        "XC60": [
            {"name": "Mk1", "year_from": 2008, "year_to": 2017},
            {"name": "Mk2", "year_from": 2017, "year_to": 2026},
        ],
        "S60": [
            {"name": "Mk1", "year_from": 2000, "year_to": 2009},
            {"name": "Mk2", "year_from": 2010, "year_to": 2018},
            {"name": "Mk3", "year_from": 2018, "year_to": 2026},
        ],
    },
    "Mini": {
        "Cooper": [
            {"name": "R50", "year_from": 2000, "year_to": 2006},
            {"name": "R56", "year_from": 2006, "year_to": 2013},
            {"name": "F56", "year_from": 2013, "year_to": 2021},
            {"name": "Mk4", "year_from": 2021, "year_to": 2026},
        ],
        "Countryman": [
            {"name": "R60", "year_from": 2010, "year_to": 2016},
            {"name": "F60", "year_from": 2017, "year_to": 2024},
        ],
    },
    "Tesla": {
        "Model 3": [
            {"name": "Gen1", "year_from": 2017, "year_to": 2023},
            {"name": "Highland", "year_from": 2023, "year_to": 2026},
        ],
        "Model X": [
            {"name": "Gen1", "year_from": 2015, "year_to": 2021},
            {"name": "Refresh", "year_from": 2021, "year_to": 2026},
        ],
        "Model S": [
            {"name": "Gen1", "year_from": 2012, "year_to": 2021},
            {"name": "Refresh", "year_from": 2021, "year_to": 2026},
        ],
        "Model Y": [
            {"name": "Gen1", "year_from": 2020, "year_to": 2024},
            {"name": "Juniper", "year_from": 2024, "year_to": 2026},
        ],
    },
    "Smart": {
        "ForTwo": [
            {"name": "W450", "year_from": 1998, "year_to": 2007},
            {"name": "W451", "year_from": 2007, "year_to": 2014},
            {"name": "W453", "year_from": 2014, "year_to": 2023},
        ],
    },
    "Cupra": {
        "Born": [{"name": "Gen1", "year_from": 2021, "year_to": 2026}],
        "Formentor": [{"name": "Gen1", "year_from": 2020, "year_to": 2026}],
    },
    "DS": {
        "DS7": [{"name": "Gen1", "year_from": 2017, "year_to": 2024}],
    },
    "Opel": {
        "Astra": [
            {"name": "F", "year_from": 1991, "year_to": 1998},
            {"name": "G", "year_from": 1998, "year_to": 2004},
            {"name": "H", "year_from": 2004, "year_to": 2009},
            {"name": "J", "year_from": 2009, "year_to": 2015},
            {"name": "K", "year_from": 2015, "year_to": 2021},
            {"name": "L", "year_from": 2021, "year_to": 2026},
        ],
        "Corsa": [
            {"name": "B", "year_from": 1993, "year_to": 2000},
            {"name": "C", "year_from": 2000, "year_to": 2006},
            {"name": "D", "year_from": 2006, "year_to": 2014},
            {"name": "E", "year_from": 2014, "year_to": 2019},
            {"name": "F", "year_from": 2019, "year_to": 2026},
        ],
        "Vectra": [
            {"name": "A", "year_from": 1988, "year_to": 1995},
            {"name": "B", "year_from": 1995, "year_to": 2002},
            {"name": "C", "year_from": 2002, "year_to": 2008},
        ],
    },
    "Renault": {
        "Clio": [
            {"name": "Mk1", "year_from": 1990, "year_to": 1998},
            {"name": "Mk2", "year_from": 1998, "year_to": 2005},
            {"name": "Mk3", "year_from": 2005, "year_to": 2012},
            {"name": "Mk4", "year_from": 2012, "year_to": 2019},
            {"name": "Mk5", "year_from": 2019, "year_to": 2026},
        ],
        "Mégane": [
            {"name": "Mk1", "year_from": 1995, "year_to": 2002},
            {"name": "Mk2", "year_from": 2002, "year_to": 2008},
            {"name": "Mk3", "year_from": 2008, "year_to": 2015},
            {"name": "Mk4", "year_from": 2016, "year_to": 2023},
        ],
        "Captur": [
            {"name": "Mk1", "year_from": 2013, "year_to": 2019},
            {"name": "Mk2", "year_from": 2019, "year_to": 2026},
        ],
        "Twingo": [
            {"name": "Mk1", "year_from": 1993, "year_to": 2007},
            {"name": "Mk2", "year_from": 2007, "year_to": 2014},
            {"name": "Mk3", "year_from": 2014, "year_to": 2024},
        ],
        "Scénic": [
            {"name": "Mk1", "year_from": 1996, "year_to": 2003},
            {"name": "Mk2", "year_from": 2003, "year_to": 2009},
            {"name": "Mk3", "year_from": 2009, "year_to": 2016},
            {"name": "Mk4", "year_from": 2016, "year_to": 2024},
        ],
        "Zoe": [
            {"name": "Gen1", "year_from": 2012, "year_to": 2019},
            {"name": "Gen2", "year_from": 2019, "year_to": 2024},
        ],
        "Modus": [{"name": "Gen1", "year_from": 2004, "year_to": 2012}],
        "4": [{"name": "Gen1", "year_from": 1961, "year_to": 1992}],
        "5": [
            {"name": "Gen1", "year_from": 1972, "year_to": 1985},
            {"name": "Supercinq", "year_from": 1984, "year_to": 1996},
        ],
    },
    "Peugeot": {
        "107": [{"name": "Gen1", "year_from": 2005, "year_to": 2014}],
        "208": [
            {"name": "Mk1", "year_from": 2012, "year_to": 2019},
            {"name": "Mk2", "year_from": 2019, "year_to": 2026},
        ],
        "308": [
            {"name": "Mk1", "year_from": 2007, "year_to": 2013},
            {"name": "Mk2", "year_from": 2013, "year_to": 2021},
            {"name": "Mk3", "year_from": 2021, "year_to": 2026},
        ],
        "307": [{"name": "Gen1", "year_from": 2001, "year_to": 2008}],
        "405": [{"name": "Gen1", "year_from": 1987, "year_to": 1997}],
        "407": [{"name": "Gen1", "year_from": 2004, "year_to": 2011}],
        "508": [
            {"name": "Mk1", "year_from": 2010, "year_to": 2018},
            {"name": "Mk2", "year_from": 2018, "year_to": 2026},
        ],
        "3008": [
            {"name": "Mk1", "year_from": 2008, "year_to": 2016},
            {"name": "Mk2", "year_from": 2016, "year_to": 2024},
        ],
        "5008": [
            {"name": "Mk1", "year_from": 2009, "year_to": 2017},
            {"name": "Mk2", "year_from": 2017, "year_to": 2024},
        ],
    },
    "Fiat": {
        "Panda": [
            {"name": "Mk1", "year_from": 1980, "year_to": 2003},
            {"name": "Mk2", "year_from": 2003, "year_to": 2012},
            {"name": "Mk3", "year_from": 2012, "year_to": 2024},
        ],
        "Punto": [
            {"name": "Mk1", "year_from": 1993, "year_to": 1999},
            {"name": "Mk2", "year_from": 1999, "year_to": 2010},
            {"name": "Grande Punto", "year_from": 2005, "year_to": 2018},
        ],
        "Freemont": [{"name": "Gen1", "year_from": 2011, "year_to": 2016}],
    },
    "Ford": {
        "KA": [
            {"name": "Mk1", "year_from": 1996, "year_to": 2008},
            {"name": "Mk2", "year_from": 2008, "year_to": 2016},
            {"name": "Ka+", "year_from": 2016, "year_to": 2021},
        ],
        "Mondeo SW": [
            {"name": "Mk3", "year_from": 2000, "year_to": 2007},
            {"name": "Mk4", "year_from": 2007, "year_to": 2014},
            {"name": "Mk5", "year_from": 2014, "year_to": 2022},
        ],
    },
    "Honda": {
        "Jazz": [
            {"name": "Mk1", "year_from": 2001, "year_to": 2008},
            {"name": "Mk2", "year_from": 2008, "year_to": 2013},
            {"name": "Mk3", "year_from": 2014, "year_to": 2020},
            {"name": "Mk4", "year_from": 2020, "year_to": 2026},
        ],
    },
    "Hyundai": {
        "i20": [
            {"name": "Mk1", "year_from": 2008, "year_to": 2014},
            {"name": "Mk2", "year_from": 2014, "year_to": 2020},
            {"name": "Mk3", "year_from": 2020, "year_to": 2026},
        ],
        "i30": [
            {"name": "Mk1", "year_from": 2007, "year_to": 2011},
            {"name": "Mk2", "year_from": 2011, "year_to": 2017},
            {"name": "Mk3", "year_from": 2017, "year_to": 2026},
        ],
        "Tucson": [
            {"name": "Mk1", "year_from": 2004, "year_to": 2009},
            {"name": "Mk2", "year_from": 2009, "year_to": 2015},
            {"name": "Mk3", "year_from": 2015, "year_to": 2020},
            {"name": "Mk4", "year_from": 2020, "year_to": 2026},
        ],
    },
    "Mazda": {
        "3": [
            {"name": "BK", "year_from": 2003, "year_to": 2009},
            {"name": "BL", "year_from": 2009, "year_to": 2013},
            {"name": "BM", "year_from": 2013, "year_to": 2019},
            {"name": "BP", "year_from": 2019, "year_to": 2026},
        ],
        "626": [
            {"name": "GD", "year_from": 1987, "year_to": 1992},
            {"name": "GE", "year_from": 1992, "year_to": 1997},
            {"name": "GF", "year_from": 1997, "year_to": 2002},
        ],
        "CX-5": [
            {"name": "Mk1", "year_from": 2012, "year_to": 2017},
            {"name": "Mk2", "year_from": 2017, "year_to": 2026},
        ],
    },
    "Nissan": {
        "Juke": [
            {"name": "F15", "year_from": 2010, "year_to": 2019},
            {"name": "F16", "year_from": 2019, "year_to": 2026},
        ],
        "Pathfinder": [
            {"name": "R51", "year_from": 2004, "year_to": 2012},
            {"name": "R52", "year_from": 2012, "year_to": 2020},
        ],
        "Pulsar": [{"name": "C13", "year_from": 2014, "year_to": 2018}],
    },
    "Porsche": {
        "Cayenne": [
            {"name": "E1", "year_from": 2002, "year_to": 2010},
            {"name": "E2", "year_from": 2010, "year_to": 2018},
            {"name": "E3", "year_from": 2018, "year_to": 2026},
        ],
        "Panamera": [
            {"name": "970", "year_from": 2009, "year_to": 2016},
            {"name": "971", "year_from": 2016, "year_to": 2026},
        ],
        "Taycan": [{"name": "Gen1", "year_from": 2019, "year_to": 2026}],
        "924": [{"name": "Gen1", "year_from": 1976, "year_to": 1988}],
        "Macan": [
            {"name": "Mk1", "year_from": 2014, "year_to": 2024},
            {"name": "Mk2", "year_from": 2024, "year_to": 2026},
        ],
    },
    "BMW": {
        "M3": [
            {"name": "E30", "year_from": 1986, "year_to": 1991},
            {"name": "E36", "year_from": 1992, "year_to": 1999},
            {"name": "E46", "year_from": 2000, "year_to": 2006},
            {"name": "E90", "year_from": 2007, "year_to": 2013},
            {"name": "F80", "year_from": 2014, "year_to": 2018},
            {"name": "G80", "year_from": 2021, "year_to": 2026},
        ],
        "X4": [
            {"name": "F26", "year_from": 2014, "year_to": 2018},
            {"name": "G02", "year_from": 2018, "year_to": 2026},
        ],
    },
    "Toyota": {
        "Aygo": [
            {"name": "Mk1", "year_from": 2005, "year_to": 2014},
            {"name": "Mk2", "year_from": 2014, "year_to": 2023},
        ],
        "RAV4": [
            {"name": "Mk3", "year_from": 2005, "year_to": 2012},
            {"name": "Mk4", "year_from": 2012, "year_to": 2018},
            {"name": "Mk5", "year_from": 2018, "year_to": 2026},
        ],
    },
    "Mitsubishi": {
        "Outlander": [
            {"name": "Mk2", "year_from": 2006, "year_to": 2012},
            {"name": "Mk3", "year_from": 2012, "year_to": 2021},
            {"name": "Mk4", "year_from": 2021, "year_to": 2026},
        ],
        "Space Star": [
            {"name": "Mk1", "year_from": 1998, "year_to": 2005},
            {"name": "Mk2", "year_from": 2012, "year_to": 2026},
        ],
    },
    "Jaguar": {
        "E-Pace": [{"name": "Gen1", "year_from": 2017, "year_to": 2024}],
        "I-Pace": [{"name": "Gen1", "year_from": 2018, "year_to": 2025}],
        "F-Pace": [{"name": "Gen1", "year_from": 2016, "year_to": 2026}],
    },
    "Jeep": {
        "Compass": [
            {"name": "MK49", "year_from": 2006, "year_to": 2017},
            {"name": "MP", "year_from": 2017, "year_to": 2026},
        ],
    },
    "Land Rover": {
        "Defender": [
            {"name": "Mk1", "year_from": 1983, "year_to": 2016},
            {"name": "L663", "year_from": 2019, "year_to": 2026},
        ],
        "Range Rover Evoque": [
            {"name": "L538", "year_from": 2011, "year_to": 2019},
            {"name": "L551", "year_from": 2019, "year_to": 2026},
        ],
    },
    "Volkswagen": {
        "Crafter": [
            {"name": "Mk1", "year_from": 2006, "year_to": 2016},
            {"name": "Mk2", "year_from": 2017, "year_to": 2026},
        ],
    },
    "Audi": {
        "A1": [
            {"name": "8X", "year_from": 2010, "year_to": 2018},
            {"name": "GB", "year_from": 2018, "year_to": 2026},
        ],
        "A3": [
            {"name": "8L", "year_from": 1996, "year_to": 2003},
            {"name": "8P", "year_from": 2003, "year_to": 2012},
            {"name": "8V", "year_from": 2012, "year_to": 2020},
            {"name": "8Y", "year_from": 2020, "year_to": 2026},
        ],
        "A4": [
            {"name": "B5", "year_from": 1994, "year_to": 2001},
            {"name": "B6", "year_from": 2001, "year_to": 2004},
            {"name": "B7", "year_from": 2004, "year_to": 2008},
            {"name": "B8", "year_from": 2008, "year_to": 2015},
            {"name": "B9", "year_from": 2015, "year_to": 2026},
        ],
        "A5": [
            {"name": "8T", "year_from": 2007, "year_to": 2016},
            {"name": "F5", "year_from": 2016, "year_to": 2026},
        ],
        "Q3": [
            {"name": "8U", "year_from": 2011, "year_to": 2018},
            {"name": "F3", "year_from": 2018, "year_to": 2026},
        ],
        "Q5": [
            {"name": "8R", "year_from": 2008, "year_to": 2017},
            {"name": "FY", "year_from": 2017, "year_to": 2026},
        ],
    },
    "Mercedes-Benz": {
        "CLA-Class": [
            {"name": "C117", "year_from": 2013, "year_to": 2019},
            {"name": "C118", "year_from": 2019, "year_to": 2026},
        ],
        "CLC-Class": [{"name": "CL203", "year_from": 2008, "year_to": 2011}],
        "GLA-Class": [
            {"name": "X156", "year_from": 2013, "year_to": 2019},
            {"name": "H247", "year_from": 2019, "year_to": 2026},
        ],
        "GLB-Class": [{"name": "X247", "year_from": 2019, "year_to": 2026}],
        "GLC-Class": [
            {"name": "X253", "year_from": 2015, "year_to": 2022},
            {"name": "X254", "year_from": 2022, "year_to": 2026},
        ],
    },
    "Kia": {
        "Ceed": [
            {"name": "Mk1", "year_from": 2006, "year_to": 2012},
            {"name": "Mk2", "year_from": 2012, "year_to": 2018},
            {"name": "Mk3", "year_from": 2018, "year_to": 2026},
        ],
    },
    "Škoda": {
        "Fabia": [
            {"name": "Mk1", "year_from": 1999, "year_to": 2007},
            {"name": "Mk2", "year_from": 2007, "year_to": 2014},
            {"name": "Mk3", "year_from": 2014, "year_to": 2021},
            {"name": "Mk4", "year_from": 2021, "year_to": 2026},
        ],
        "Superb": [
            {"name": "Mk1", "year_from": 2001, "year_to": 2008},
            {"name": "Mk2", "year_from": 2008, "year_to": 2015},
            {"name": "Mk3", "year_from": 2015, "year_to": 2024},
        ],
    },
}


# =========================================================================
# Provider: DBpedia SPARQL
# =========================================================================

_DBPEDIA_ENDPOINT = "https://dbpedia.org/sparql"
_DBPEDIA_QUERY = """\
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?genLabel ?mfgLabel ?sy ?ey WHERE {
  ?gen a dbo:Automobile .
  ?gen dbo:manufacturer ?mfg .
  ?gen dbo:productionStartYear ?sy .
  OPTIONAL { ?gen dbo:productionEndYear ?ey . }
  ?gen rdfs:label ?genLabel . FILTER(LANG(?genLabel) = "en")
  ?mfg rdfs:label ?mfgLabel . FILTER(LANG(?mfgLabel) = "en")
} ORDER BY ?mfgLabel ?genLabel
"""

_PAREN_RE = re.compile(r"^(.+?)\s*\(([^)]+)\)\s*$")
_MK_RE = re.compile(r"^(.+?)\s+(Mk\s*\.?\s*\d+)\s*$", re.I)
_ORDINAL_RE = re.compile(
    r"^(.+?)\s*\((first|second|third|fourth|fifth|sixth|seventh|"
    r"eighth|ninth|tenth|eleventh|twelfth)\s+generation.*\)\s*$", re.I,
)
_ORDINAL_MAP = {
    "first": "I", "second": "II", "third": "III", "fourth": "IV",
    "fifth": "V", "sixth": "VI", "seventh": "VII", "eighth": "VIII",
    "ninth": "IX", "tenth": "X", "eleventh": "XI", "twelfth": "XII",
}


def _parse_dbpedia_label(label: str, brand: str) -> tuple[str, str] | None:
    name = label
    for prefix in (brand, brand.split()[0]):
        if name.startswith(prefix + " "):
            name = name[len(prefix) + 1:]
            break
    m = _ORDINAL_RE.match(name)
    if m:
        return m.group(1).strip(), _ORDINAL_MAP[m.group(2).lower()]
    m = _PAREN_RE.match(name)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = _MK_RE.match(name)
    if m:
        return m.group(1).strip(), m.group(2).replace(" ", "").replace(".", "")
    return None


def _fetch_dbpedia() -> dict[str, dict[str, list]]:
    logger.info("Provider: DBpedia — fetching...")
    body = urllib.parse.urlencode({"query": _DBPEDIA_QUERY}).encode()
    req = urllib.request.Request(
        _DBPEDIA_ENDPOINT, data=body,
        headers={
            "Accept": "application/sparql-results+json",
            "User-Agent": "olx-car-parser/1.0",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = json.loads(resp.read())

    seen: set[tuple] = set()
    result: dict[str, dict[str, list]] = {}
    for row in raw["results"]["bindings"]:
        raw_brand = row["mfgLabel"]["value"]
        brand = _BRAND_NORMALIZE.get(raw_brand, raw_brand)
        gen_label = row["genLabel"]["value"]
        try:
            year_from = int(row["sy"]["value"])
        except (ValueError, KeyError):
            continue
        year_to_raw = row.get("ey", {}).get("value")
        try:
            year_to = int(year_to_raw) if year_to_raw else 2026
        except ValueError:
            year_to = 2026
        if year_from < 1950 or year_from > 2030:
            continue
        parsed = _parse_dbpedia_label(gen_label, brand)
        if not parsed:
            parsed = _parse_dbpedia_label(gen_label, raw_brand)
        if not parsed:
            continue
        series, gen_name = parsed
        key = (brand, series, gen_name)
        if key in seen:
            continue
        seen.add(key)
        result.setdefault(brand, {}).setdefault(series, []).append({
            "name": gen_name, "year_from": year_from, "year_to": year_to,
        })

    _fix_year_ranges(result)
    total = sum(len(g) for m in result.values() for g in m.values())
    logger.info("DBpedia: %d brands, %d generations", len(result), total)
    return result


# =========================================================================
# Helpers
# =========================================================================

def _fix_year_ranges(data: dict):
    for brand_data in data.values():
        for model_gens in brand_data.values():
            model_gens.sort(key=lambda g: g["year_from"])
            for i, gen in enumerate(model_gens):
                if gen["year_to"] <= gen["year_from"]:
                    if i + 1 < len(model_gens):
                        gen["year_to"] = model_gens[i + 1]["year_from"] - 1
                    else:
                        gen["year_to"] = 2026


def _merge(base: dict, extra: dict):
    for brand, models in extra.items():
        for model, gens in models.items():
            if model not in base.get(brand, {}):
                base.setdefault(brand, {})[model] = gens


def _lookup_gens(data: dict, brand: str, model: str) -> list | None:
    # Try original brand, then canonical alias
    for b in (brand, _BRAND_LOOKUP.get(brand, brand)):
        gens = data.get(b, {}).get(model)
        if gens:
            return gens
        alias = _MODEL_ALIASES.get(b, {}).get(model)
        if alias:
            gens = data.get(b, {}).get(alias)
            if gens:
                return gens
    return None


# =========================================================================
# Public API
# =========================================================================

def fetch_generations() -> dict:
    """Fetch from DBpedia, merge with static fallback data."""
    result = {}

    try:
        _merge(result, _fetch_dbpedia())
    except Exception as e:
        logger.warning("DBpedia failed: %s", e)

    _merge(result, _STATIC_GENERATIONS)

    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    total = sum(len(g) for m in result.values() for g in m.values())
    logger.info("Total: %d brands, %d generations saved to %s", len(result), total, _CACHE_PATH)

    global _generations
    _generations = result
    return result


def _cache_is_stale() -> bool:
    if not _CACHE_PATH.exists():
        return True
    return (time.time() - _CACHE_PATH.stat().st_mtime) > _CACHE_MAX_AGE


def load_generations() -> dict:
    """Load generations, auto-fetching if cache is stale (daily)."""
    global _generations
    if _generations is not None:
        return _generations
    if _cache_is_stale():
        try:
            _generations = fetch_generations()
            return _generations
        except Exception as e:
            logger.warning("Fetch failed, using cache: %s", e)
    if _CACHE_PATH.exists():
        with open(_CACHE_PATH, encoding="utf-8") as f:
            _generations = json.load(f)
    else:
        _generations = {}
    return _generations


def get_generation(brand: str, model: str, year: int | None) -> str | None:
    """Return generation name for a given car, or None if unknown."""
    if not year:
        return None
    data = load_generations()
    gens = _lookup_gens(data, brand, model)
    if not gens:
        return None
    for g in gens:
        if g["year_from"] <= year <= g["year_to"]:
            return g["name"]
    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = fetch_generations()
    total = sum(len(g) for m in data.values() for g in m.values())
    print(f"\nTotal: {len(data)} brands, {total} generations")
