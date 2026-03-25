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


Install pre-compiled package from conda-forge
---------------------------------------------

We strongly recommend setting up a new conda environment for using ``libparallelproj``.
This ensures that the correct dependencies are installed and prevents conflicts with other packages in existing environments (especially when using CUDA).

C-API only
^^^^^^^^^^

.. tab-set::

	.. tab-item:: mamba

		.. code-block:: bash

			mamba create -n my_new_env -c conda-forge "libparallelproj>=2"

	.. tab-item:: conda

		.. code-block:: bash

			conda create -n my_new_env -c conda-forge "libparallelproj>=2"

.. note::

	On systems with a CUDA device, the conda-forge solver should select a CUDA build of ``libparallelproj`` automatically.
	CUDA 12 and 13 builds are available, and the solver should select the correct build.
	In case you want to force a specific build (e.g. CUDA 12 vs CUDA 13 vs CPU), you can use

.. tab-set::

	.. tab-item:: CUDA 12.9

		.. code-block:: bash

			mamba create -n my_new_env -c conda-forge cuda-version=12.9 "libparallelproj>=2"

	.. tab-item:: CUDA 13.x

		.. code-block:: bash

			mamba create -n my_new_env -c conda-forge cuda-version=13 "libparallelproj>=2"

	.. tab-item:: CPU

		.. code-block:: bash

			mamba create -n my_new_env -c conda-forge "libparallelproj>=2=cpu*"


Python API (includes C-API)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install the conda-forge package ``parallelproj-core``:

.. tab-set::

	.. tab-item:: mamba

		.. code-block:: bash

			mamba create -n my_new_env -c conda-forge parallelproj-core

	.. tab-item:: conda

		.. code-block:: bash

			conda create -n my_new_env -c conda-forge parallelproj-core

.. note::

	``parallelproj-core`` depends on ``libparallelproj``. In case the solver wants to install the wrong build (e.g. CUDA vs CPU), use the note above to force the correct build of ``libparallelproj`` by adding the it as a dependency in the command above, e.g.:

.. tab-set::

	.. tab-item:: CUDA 12.9

		.. code-block:: bash

			mamba create -n my_new_env -c conda-forge cuda-version=12.9 parallelproj-core

	.. tab-item:: CUDA 13.x

		.. code-block:: bash

			mamba create -n my_new_env -c conda-forge cuda-version=13 parallelproj-core

	.. tab-item:: CPU

		.. code-block:: bash

			mamba create -n my_new_env -c conda-forge "libparallelproj=*=cpu*" parallelproj-core



Compile from source
-------------------

For source builds, follow the instructions in the `GitHub project README <https://github.com/KUL-recon-lab/libparallelproj/blob/main/README.md>`_
