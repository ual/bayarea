FROM continuumio/miniconda3

RUN conda create -n usim python=3.6
RUN echo "source activate usim" > ~/.bashrc
ENV PATH /opt/conda/envs/usim/bin:$PATH
ENV AWS_ACCESS_KEY_ID
ENV AWS_ACCESS_KEY_ID

WORKDIR /home
RUN git clone https://github.com/ual/activitysynth.git
	# && conda install s3fs -c conda-forge \
	# && conda install -c conda-forge fastparquet
