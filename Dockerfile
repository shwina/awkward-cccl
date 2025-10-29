FROM rapidsai/devcontainers:25.10-cpp-cuda12.9

# Build arguments
ARG GIT_BRANCH_NAME=main

# Copy requirements file
COPY requirements-cu12.txt /tmp/requirements.txt

# Install Python packages from requirements.txt
RUN pip install --root-user-action ignore --no-cache-dir -r /tmp/requirements.txt

# Set working directory
WORKDIR /awkward-cccl

# Copy notebook and helper files
COPY Dockerfile /awkward-cccl/Dockerfile
COPY awkward-cccl.ipynb /awkward-cccl/
COPY ak_helpers.py /awkward-cccl/
COPY mass_awkward.py /awkward-cccl/
COPY mass_cuda_compute.py /awkward-cccl/
COPY images/ /awkward-cccl/images/

RUN ls -la /awkward-cccl/

# Expose Jupyter port
EXPOSE 8888

# Set up Jupyter to allow connections from any IP
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]

