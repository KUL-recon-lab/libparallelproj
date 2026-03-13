Installation
============

.. note::

   **About version 2.0.0 and earlier releases**

   ``libparallelproj`` version 2.0.0 introduced a significant restructuring of the project. Versions earlier than 2.0.0 came from a different repository called ``parallelproj``, which provided a mixed package containing both:

   - A minimal C and Python API for core 3D projection functions
   - Higher-level Python tools for scanner geometry, sinogram handling, and image reconstruction

   Starting with version 2.0.0, ``libparallelproj`` focuses exclusively on the core projection functions and their APIs. The ``parallelproj-core`` package provides the minimal Python API to the core projectors using the stable Python ABI (python-abi3), ensuring compatibility across Python versions, without the need for recompilation.

   Future versions of the existing ``parallelproj`` python package (v>= 2.0) will build upon ``parallelproj-core`` as the standard API to the projectors, providing higher-level tools for scanner geometry, sinogram handling, and image reconstruction.

   The project consolidation provides a cleaner separation of concerns and improved maintainability. Version 2.0.0 and later are the recommended versions for all new projects and updates.


Option 1 (recommended): install pre-compiled package from conda-forge
---------------------------------------------------------------------

C-API only
^^^^^^^^^^

Install the conda-forge package ``libparallelproj`` (version >= 2.0.0):

.. tab-set::

	.. tab-item:: mamba

		.. code-block:: bash

			mamba install -c conda-forge libparallelproj=2

	.. tab-item:: conda

		.. code-block:: bash

			conda install -c conda-forge libparallelproj=2

.. note::

	On systems with a CUDA device, the conda-forge solver should select a CUDA build of ``libparallelproj`` automatically.
	At least ``cuda >= 12.9`` is required for CUDA builds, and this and newer CUDA versions are available on conda-forge.
	To force a specific build, use one of the following patterns:

	.. code-block:: bash

		mamba install -c conda-forge "libparallelproj=2=cuda12*"
		mamba install -c conda-forge "libparallelproj=2=cuda13*"
		mamba install -c conda-forge "libparallelproj=2=*cpu*"

.. warning::

	In existing conda environments that contain older CUDA versions (``cuda < 12.9``), the CUDA build of ``libparallelproj`` cannot be installed.

Python API (includes C-API)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install the conda-forge package ``parallelproj-core``:

.. tab-set::

	.. tab-item:: mamba

		.. code-block:: bash

			mamba install -c conda-forge "parallelproj-core"

	.. tab-item:: conda

		.. code-block:: bash

			conda install -c conda-forge "parallelproj-core"


Option 2: compile from source
-----------------------------

For source builds, follow the instructions in the `GitHub project README <https://github.com/KUL-recon-lab/libparallelproj/blob/main/README.md>`_
