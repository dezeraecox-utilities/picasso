Picasso
=======
.. image:: https://readthedocs.org/projects/picassosr/badge/?version=latest
   :target: https://picassosr.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://github.com/jungmannlab/picasso/workflows/CI/badge.svg
   :target: https://github.com/jungmannlab/picasso/workflows/CI/badge.svg
   :alt: CI

.. image:: http://img.shields.io/badge/DOI-10.1038/nprot.2017.024-52c92e.svg
   :target: https://doi.org/10.1038/nprot.2017.024
   :alt: CI

.. image:: https://static.pepy.tech/personalized-badge/picassosr?period=total&units=international_system&left_color=black&right_color=brightgreen&left_text=Downloads
 :target: https://pepy.tech/project/picassosr

.. image:: main_render.png
   :scale: 100 %
   :alt: UML Render view

A collection of tools for painting super-resolution images. The Picasso software is complemented by our `Nature Protocols publication <https://www.nature.com/nprot/journal/v12/n6/abs/nprot.2017.024.html>`__.
A comprehensive documentation can be found here: `Read the Docs <https://picassosr.readthedocs.io/en/latest/?badge=latest>`__.


Picasso 0.6.0
-------------

RESI (Resolution Enhancement by Sequential Imaging)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
RESI dialog added to Picasso Render, allowing for substatial boost in spatial resolution (*to be published*).

Photon conversion update
~~~~~~~~~~~~~~~~~~~~~~~~
The formula for conversion of raw data to photons was changed, resulting in different numbers of photons and thus **affecting the localization precision** accordingly.

Until version *0.5.7*, the formula was: 

*(RAW_DATA - BASELINE) x SENSITIVITY / (GAIN x QE)*, where QE is quantum efficiency of the camera. 

In Picasso *0.6.0* it was changed to:

*(RAW_DATA - BASELINE) x SENSITIVITY / GAIN*

**i.e., quantum effiency was removed.** Thus, the estimate of the localization precision better approximates the true precision.


For backward compatibility, quantum efficiency will be kept in Picasso Localize, however, it will have no effect on the new photon conversion formula.

Picasso 0.5.0
-------------
Picasso has introduced many changes, including 3D rotation window and a new clustering algorithm in Render and reading of .nd2 files in Localize. Please check the `changelog <https://github.com/jungmannlab/picasso/blob/master/changelog.rst>`_ to see all modifications.

Picasso 0.4.0
-------------
Picasso now has a server-based workflow management-system. Check out `here <https://picassosr.readthedocs.io/en/latest/server.html>`__.


Installation
------------

Check out the `Picasso release page <https://github.com/jungmannlab/picasso/releases/>`__ to download and run the latest compiled one-click installer for Windows. Here you will also find the Nature Protocols legacy version. 

For the platform-independent usage of Picasso (e.g., with Linux and Mac Os X), please follow the advanced installation instructions below.

Other installation modes (Python 3.8)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As an alternative to the stand-alone program for end-users, Picasso can be installed as a Python package. This is the preferred option to use Picasso’s internal routines in custom Python programs. Those can be imported by running, for example, ``from picasso import io`` (see the "Example usage" tab below) to use input/output functions from Picasso. For windows, it is still possible to use Picasso as an end-user by creating the respective shortcuts. This allows Picasso to be used on the same system by both programmers and end-users.

Via PyPI
^^^^^^^^

1. Open the console/terminal and create a new conda environment: ``conda create --name picasso python=3.8``
2. Activate the environment: ``conda activate picasso``.
3. Install Picasso package using: ``pip install picassosr``.
4. You can now run any Picasso function directly from the console/terminal by running: ``picasso render``, ``picasso localize``, etc.

For Developers
^^^^^^^^^^^^^^

If you wish to use your local version of Picasso with your own modifications:

1. Open the console/terminal and create a new conda environment: ``conda create --name picasso python=3.8``
2. Activate the environment: ``conda activate picasso``.
3. Change to the directory of choice using ``cd``.
4. Clone this GitHub repository by running ``git clone https://github.com/jungmannlab/picasso``. Alternatively, `download <https://github.com/jungmannlab/picasso/archive/master.zip>`__ the zip file and unzip it.
5. Open the Picasso directory: ``cd picasso``.
6. You can modify Picasso code from here.

*Windows*
'''''''''

7. If you wish to create a *local* Picasso package to use it in other Python scripts (that includes your changes), run ``python setup.py install``. 
8. You can now run any Picasso function directly from the console/terminal by running: ``picasso render``, ``picasso localize``, etc.
9. Remember that in order to update changes in Picasso code, you need to repeat step 7.

*Mac*
'''''

Currently, Picasso does not support package creation on Mac OS. If you wish to run your modified Picasso code, simply go to your ``picasso`` directory and run ``python -m picasso render``, ``python -m picasso localize``, etc.

Optional packages
^^^^^^^^^^^^^^^^^

Regardless of whether Picasso was installed via PyPI or by cloning the GitHub repository, some packages may be additionally installed to allow extra functionality:

- ``pip install hdbscan`` for clustering with `HDBSCAN <https://hdbscan.readthedocs.io/en/latest/index.html>`__.
- ``pip install pyinstaller`` if you plan to additionally compile your own installer with `Pyinstaller <https://pyinstaller.org/en/stable/>`__.

To enable GPU fitting, follow instructions on `Gpufit <https://github.com/gpufit/Gpufit>`__ to install the Gpufit python library in your conda environment. In practice, this means downloading the zipfile and installing the Python wheel. Picasso Localize will automatically import the library if present and enables a checkbox for GPU fitting when selecting the LQ-Method.

Updating
^^^^^^^^

If Picasso was installed from PyPI, run the following command:

``pip install --upgrade picassosr``

If Picasso was cloned from the GitHub repository, use the following commands:

1. Move to the ``picasso`` folder with the terminal, activate environment.
2. Update with git: ``git pull``.
3. Update the environment: ``pip install --upgrade -r requirements.txt``.
4. (*Windows only*)Run installation ``python setup.py install``.

Creating shortcuts on Windows (*optional*)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the PowerShell script “createShortcuts.ps1” in the gui directory. This should be doable by right-clicking on the script and choosing “Run with PowerShell”. Alternatively, run the command
``powershell ./createShortcuts.ps1`` in the command line. Use the generated shortcuts in the top level directory to start GUI components. Users can drag these shortcuts to their Desktop, Start Menu or Task Bar.

Example Usage
-------------

Besides using the GUI, you can use picasso like any other Python module. Consider the following example:::

  from picasso.picasso import io, postprocess

  path = 'testdata_locs.hdf5'
  locs, info = io.load_locs(path)
  # Link localizations and calcualte dark times
  linked_locs = postprocess.link(picked_locs, info, r_max=0.05, max_dark_time=1)
  linked_locs_dark = postprocess.compute_dark_times(linked_locs)

  print('Average bright time {:.2f} frames'.format(np.mean(linked_locs_dark.n)))
  print('Average dark time {:.2f} frames'.format(np.mean(linked_locs_dark.dark)))

This codeblock loads data from testdata_locs and uses the postprocess functions programmatically.

Jupyter Notebooks
-----------------

Check picasso/samples/ for Jupyter Notebooks that show how to interact with the Picasso codebase.

Contributing
------------

If you have a feature request or a bug report, please post it as an issue on the GitHub issue tracker. If you want to contribute, put a PR for it. You can find more guidelines for contributing `here <https://github.com/jungmannlab/picasso/blob/master/CONTRIBUTING.rst>`__. I will gladly guide you through the codebase and credit you accordingly. Additionally, you can check out the ``Projects``-page on GitHub.  You can also contact me via picasso@jungmannlab.org.

Contributions & Copyright
-------------------------

| Contributors: Joerg Schnitzbauer, Maximilian Strauss, Rafal Kowalewski, Adrian Przybylski, Andrey Aristov, Hiroshi Sasaki, Alexander Auer, Johanna Rahm
| Copyright (c) 2015-2019 Jungmann Lab, Max Planck Institute of Biochemistry
| Copyright (c) 2020-2021 Maximilian Strauss
| Copyright (c) 2022 Rafal Kowalewski

Citing Picasso
--------------

If you use picasso in your research, please cite our Nature Protocols publication describing the software.

| J. Schnitzbauer*, M.T. Strauss*, T. Schlichthaerle, F. Schueder, R. Jungmann
| Super-Resolution Microscopy with DNA-PAINT
| Nature Protocols (2017). 12: 1198-1228 DOI: `https://doi.org/10.1038/nprot.2017.024 <https://doi.org/10.1038/nprot.2017.024>`__

Credits
-------

-  Design icon based on “Hexagon by Creative Stalls from the Noun
   Project”
-  Simulate icon based on “Microchip by Futishia from the Noun Project”
-  Localize icon based on “Mountains by MONTANA RUCOBO from the Noun
   Project”
-  Filter icon based on “Funnel by José Campos from the Noun Project”
-  Render icon based on “Paint Palette by Vectors Market from the Noun
   Project”
-  Average icon based on “Layers by Creative Stall from the Noun
   Project”
-  Server icon based on “Database by Nimal Raj from NounProject.com”
