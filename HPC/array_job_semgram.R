"""
This is the r code for running gender analysis for my thesis on NYU's HPC cluster.
The package being used to extract gender motifs here is semgram by Oscar Stuhler(see https://github.com/omstuhler/semgram for more details.)



library(reticulate)
Sys.unsetenv("RETICULATE_PYTHON") 
#use_virtualenv("/home/jy3440/.virtualenvs/r-reticulate", required = TRUE)
library(pryr)
library(readr)
library(spacyr)
library(dplyr)
library(semgram)
library(data.table)
library(jsonlite)
library(tm)


task_id <- Sys.getenv("SLURM_ARRAY_TASK_ID")
task_id <- as.integer(task_id)

file_name <- sprintf("/scratch/jy3440/MOTIFS/corpus_chunks/corpus_chunk_%02d.csv", task_id)

spacy_initialize()

# format input corpus 
formatting_text <- function(text) {
  # replace normal "" pairs first
  text <- gsub('""(.*?)""', '"\\1"', text)
  
  # replace all remaining "" with "
  text <- gsub('""', '"', text)
  
  # remove newlines and replace them with spaces
  text <- gsub("\n", " ", text)
  
  # replace multiple whitespaces with a single space
  text <- gsub("[ ]{2,}", " ", text)
  
  return(text)
}

process_books <- function(text) {
  chunk_size_limit <- 1000000  # set the chunk size limit
  
  # initialize an empty list to store chunks
  chunks <- list()
  
  # function to add chunk ensuring it doesn't exceed the spacy limit.
  add_chunk <- function(text, start, end) {
    if (end < nchar(text)) {
      # make sure words in chunks are complete
      while (substring(text, end + 1, end + 1) != ' ' && end > start) {
        end <- end - 1
      }
    }
    return(substring(text, start, end))
  }
  
  # check and chunk the text only if it exceeds the limit.
  if (nchar(text) > chunk_size_limit) {
    # split text into chunks 
    start_index <- 1
    repeat {
      end_index <- min(start_index + chunk_size_limit - 1, nchar(text))
      chunk <- add_chunk(text, start_index, end_index)
      chunks <- c(chunks, list(chunk))
      if (end_index == nchar(text)) break
      start_index <- end_index + 1
    }
  } else {
    chunks <- list(text)
  }
  
  # Initialize an empty list to store results from each chunk.
  result <- list()
  error_log <- list()  # To record any errors encountered
  
  for (i in seq_along(chunks)) {
    tryCatch({
      chunk <- chunks[[i]]
      # ensure spacy_parse processes the chunk within limit.
      tokens_df <- spacy_parse(chunk, dependency = TRUE)
      motifs <- extract_motifs(tokens_df, entities = c("*"), pron_as_ap = TRUE, parse_multi_token_entities = T)
      
      result[[paste("chunk", i, sep = "_")]] <- motifs
    }, error = function(e) {
      # log the error message and the chunk number
      error_log[[paste("chunk", i, sep = "_")]] <- e$message
      print(e$message)
    })
  }
  
  # combine all motifs and data frames from each chunk.
  # initialize an empty list for each category
  combined_actions <- list()
  combined_treatments <- list()
  combined_characterizations <- list()
  combined_possessions <- list()
  combined_agent_treatments <- list()
  combined_action_patients <- list()
  
  # loop thru each category of each chunk in the chunk list and combine 
  for (i in seq_along(result)) {
    combined_actions[[i]] <- result[[i]]$actions
    combined_treatments[[i]] <- result[[i]]$treatments
    combined_characterizations[[i]] <- result[[i]]$characterizations
    combined_possessions[[i]] <- result[[i]]$possessions
    combined_agent_treatments[[i]] <- result[[i]]$agent_treatments
    combined_action_patients[[i]] <- result[[i]]$action_patients
  }
  
  # combine each category
  final_actions <- do.call(rbind, combined_actions)
  final_treatments <- do.call(rbind, combined_treatments)
  final_characterizations <- do.call(rbind, combined_characterizations)
  final_possessions <- do.call(rbind, combined_possessions)
  final_agent_treatments <- do.call(rbind, combined_agent_treatments)
  final_action_patients <- do.call(rbind, combined_action_patients)
  
  # write the final list of combined data frames
  final_combined_list <- list(
    actions = final_actions,
    treatments = final_treatments,
    characterizations = final_characterizations,
    possessions = final_possessions,
    agent_treatments = final_agent_treatments,
    action_patients = final_action_patients
  )

# print once so I can monitor the progress in the .out files
  print("motifs for one work extracted")
  return(final_combined_list)
}

# set chunk size
start_row <- 0
chunk_size <- 1000
total_rows <- 5000
subchunk <- 0

# codes for processing each chunk
while (start_row < total_rows) {
  # Read a chunk of data
  corpus_chunk <- fread(file_name, skip = start_row, nrows = chunk_size)
  print("Reading one subchunk done.")

  # format one corpus chunk
  corpus_chunk[!is.na(body) & body != '', body := lapply(.SD, formatting_text), .SDcols = "body"]
  print("Formatting done for one corpus chunk.")

  # apply motif extraction
  corpus_chunk[, motif := ifelse(!is.na(body) & body != '', lapply(body, process_books), '')]
  print("Motif extraction done for one corpus chunk.")

  # monitor memory usage before writing JSON
  print(paste("Memory usage before writing JSON:", pryr::mem_used()))

  # construct the output filename and write to JSON
  output_name <- sprintf("corpus_chunks_results/motifs_%02d_%02d.json", as.integer(task_id), subchunk)
  write_json(corpus_chunk, output_name)
  
  # print memory usage after writing JSON
  print(paste("Memory usage after writing JSON:", pryr::mem_used()))

  # explicitly remove the chunk from memory and perform garbage collection
  rm(corpus_chunk)
  gc()

  # monitor memory again
  print(paste("Memory usage after gc():", pryr::mem_used()))

  # update the starting row and subchunk index for the next iteration
  start_row <- start_row + chunk_size
  subchunk <- subchunk + 1
  cat(sprintf("Processed up to row %d\n", start_row))
}
