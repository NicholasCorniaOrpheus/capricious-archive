# Capricious Archive

Inspired by the book edited by Italian scholar Paolo Procaccioli _Cinquecento capriccioso e irregolare_, this AI tool allows the used to emulatively dialogue with the transgressive corpus of the [_Accademia degli Incompresi_](https://nicholascorniaorpheus.github.io/accademia-degli-incompresi/) knowledge base.

This project is part of the artistic research PhD [_A Whirlpool of Imaginary Sounds_](https://orpheusinstituut.be/en/projects/a-whirlpool-of-imaginary-sounds) led by [Nicholas Cornia](https://orpheusinstituut.be/en/research/researchers/nicholas-cornia) in collaboration with [Thomas Ballhausen](https://www.moz.ac.at/en/people/scenography/thomas-ballhausen) and [Elena Peytchinska](https://www.elenapeytchinska.com/). It is still in development phase and the code has been made with the help of Claude Sonnet 5.5 taking into account scalability, computational and cost efficiencies, and sustainability.

# Code structure

- `app.py` runs the Streamlit code.
- `cli.py` runs statistics, style profile, and a terminal chat for prompting.
- `llm.py`
- `corpus.py` normalizes the texts into modern Italian spelling.