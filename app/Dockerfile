FROM rocker/shiny:4.0.5

# system libraries
RUN apt-get update && apt-get install -y \
    libcurl4-gnutls-dev \
    libssl-dev \
    libxml2-dev \
    libudunits2-dev \
    libproj-dev \
    libpq-dev \
    gdal-bin \
    libgdal-dev


  

# install R packages required 
RUN R -e 'install.packages(c(\
              "shiny", "httr",\
              "shinydashboard", \
              "dplyr", "tidyverse", "RPostgreSQL",\
              "texreg", "sf", "ggthemes", "lfe", "shinylogs", \
              "jsonlite", "RNifti", "shinythemes","DT",\
              "shinyWidgets", "markdown"\
            ) \
          )'


COPY . app/
EXPOSE 80

CMD ["R", "-e", "shiny::runApp('/app', host = '0.0.0.0', port = 80)"]
