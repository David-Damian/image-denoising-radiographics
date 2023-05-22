library(shiny)
library(RNifti)
library(shinythemes)
library(shinyWidgets)
library(markdown)
source("lazyr.R")
source("interactive.R")
source("animation.R")

read_img_as_array <- function(path) {
  img_raw <- RNifti::readNifti(path)
  if (length(dim(img_raw)) == 3) {
    return(img_raw[, , ])
  }
  return(img_raw[, , , ])
}

ui <- navbarPage(
  "Shiny MRI",
  theme = shinytheme("cyborg"),
  tabPanel(
    "Home",
    fluidRow(
      column(4, fileInput("your_dt", "Upload a .jpg, .png image")),
      column(1, h2("|"),
        class = "text-center",
        style = "margin-top: -5px; "
      ),
      column(3, shinyWidgets::switchInput(
        "interactive", "Interactive",
        onStatus = "success"
      ), style = "padding-top: 25px;")
    ),
    uiOutput("raster_panel")
  ),
  tabPanel(
    "About",
    includeMarkdown("about.md")
  )
)

server <- function(input, output, session) {
  options(shiny.maxRequestSize = 500 * 1024^2)

  app_dt <- reactive({
    if (is.null(input$your_dt)) {
      out <- read_img_as_array(input$demo_dt)
    } else {
      datapath <- input$your_dt$datapath
      if (tools::file_ext(datapath) == "gz") {
        datapath <- sub("gz$", "nii.gz", datapath)
        file.rename(input$your_dt$datapath, datapath)
      }
      out <- read_img_as_array(datapath)
    }
    return(out)
  })

  output$raster_panel <- renderUI({
    if (input$interactive) {
      callModule(raster3d_interactive_Module, "mri_3d", im = app_dt)
      raster3d_interactive_UI("mri_3d")
    } else {
      callModule(raster3d_animation_Module, "mri_3d", im = app_dt)
      raster3d_animation_UI("mri_3d")
    }
  })
}

shinyApp(ui, server)
