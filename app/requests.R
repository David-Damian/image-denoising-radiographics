library(httr)


post_request <- function(path) {
    library(httr)

    body <- list(
        "file" = upload_file(path)
    )
    res <- VERB("POST", url = "http://0.0.0.0:8080/uploadphoto/", body = body, encode = "multipart")
    return(res)
}
