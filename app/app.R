library(shiny)
library(RNifti)
library(shinythemes)
library(shinyWidgets)
library(markdown)
library(base64enc)
library(shinyjs)

source("requests.R")

options(shiny.maxRequestSize = 30*1024^2)

read_img_as_array <- function(path) {
  img_raw <- RNifti::readNifti(path)
  if (length(dim(img_raw)) == 3) {
    return(img_raw[, , ])
  }
  return(img_raw[, , , ])
}

ui <- navbarPage(
  "Rayos X",
  theme = shinytheme("cyborg"),
  tabPanel(
    "Home",
    fluidRow(
      column(4, fileInput("upload", "Upload a .png image", accept = "image/png")),
      column(1, h2("|"),
        class = "text-center",
        style = "margin-top: -5px; "
      )
    ),
    fluidRow(
      column(4, uiOutput("image")),
      column(1, uiOutput("image_2"))
    ),
  ),
  tabPanel(
    "About",
    includeMarkdown("about.md")
  )
)

server <- function(input, output){

  base64 <- reactive({
    inFile <- input[["upload"]]
    if(!is.null(inFile)){
      dataURI(file = inFile$datapath, mime = "image/png")
      # status <- post_request(ext)

    }
  })

  output[["image"]] <- renderUI({
    if(!is.null(base64())){
      tags$div(
        tags$img(src= base64(), width="100%"),
        style = "width: 400px;"
      )
    }
  })

  output[["image_2"]] <- renderUI({
    if(!is.null(base64())){
      b64_2 <- base64enc::dataURI(file="/app/mock/original_resized.png", mime="image/png")
      delay(1000)
      tags$div(
        tags$img(src = b64_2, width="100%"),
        style = "width: 400px;"
      )
    }
  })
}
shinyApp(ui, server)
