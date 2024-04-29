{\rtf1\ansi\ansicpg1252\cocoartf2758
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 \
#this is the shell script that works on chunks of 2000, array 1-5, with print memory checking points.\
#all the prints will be removed when I scale up the analysis.\
#this should be used with smaller chunks of data\
\
#this is for HPC cluster\
\
library(reticulate)\
Sys.unsetenv("RETICULATE_PYTHON") \
#use_virtualenv("/home/jy3440/.virtualenvs/r-reticulate", required = TRUE)\
library(pryr)\
library(readr)\
library(spacyr)\
library(dplyr)\
library(semgram)\
library(data.table)\
library(jsonlite)\
library(tm)\
\
# Assuming SLURM scheduler for the HPC cluster\
task_id <- Sys.getenv("SLURM_ARRAY_TASK_ID")\
task_id <- as.integer(task_id)\
\
file_name <- sprintf("/scratch/jy3440/MOTIFS/corpus_chunks/corpus_chunk_%02d.csv", task_id)\
corpus <- fread(file_name)\
\
# check memory\
print(paste("Memory usage after loading the corpus:", pryr::mem_used()))\
\
#Initialize spacy\
spacy_initialize()\
\
# check memory\
print(paste("Memory usage after initializing spacy:", pryr::mem_used()))\
\
formatting_text <- function(text) \{\
  # replace normal "" pairs first\
  text <- gsub('""(.*?)""', '"\\\\1"', text)\
  \
  # replace all remaining "" with "\
  text <- gsub('""', '"', text)\
  \
  # remove newlines and replace them with spaces\
  text <- gsub("\\n", " ", text)\
  \
  # replace multiple whitespaces with a single space\
  text <- gsub("[ ]\{2,\}", " ", text)\
  \
  return(text)\
\}\
\
process_books <- function(text) \{\
  chunk_size_limit <- 1000000  # set the chunk size limit\
  \
  # initialize an empty list to store chunks.\
  chunks <- list()\
  \
  # function to add chunk ensuring it doesn't exceed the spacy limit.\
  add_chunk <- function(text, start, end) \{\
    if (end < nchar(text)) \{\
      # make sure words in chunks are complete\
      while (substring(text, end + 1, end + 1) != ' ' && end > start) \{\
        end <- end - 1\
      \}\
    \}\
    return(substring(text, start, end))\
  \}\
  \
  # check and chunk the text only if it exceeds the limit.\
  if (nchar(text) > chunk_size_limit) \{\
    # split text into chunks \
    start_index <- 1\
    repeat \{\
      end_index <- min(start_index + chunk_size_limit - 1, nchar(text))\
      chunk <- add_chunk(text, start_index, end_index)\
      chunks <- c(chunks, list(chunk))\
      if (end_index == nchar(text)) break\
      start_index <- end_index + 1\
    \}\
  \} else \{\
    chunks <- list(text)\
    print(length(chunks))\
  \}\
  \
  # niitialize an empty list to store results from each chunk.\
  result <- list()\
  error_log <- list()  # To record any errors encountered\
  \
  for (i in seq_along(chunks)) \{\
    print(paste("processing chunk", i))\
    tryCatch(\{\
      chunk <- chunks[[i]]\
      # ensure spacy_parse processes the chunk within limit.\
      tokens_df <- spacy_parse(chunk, dependency = TRUE)\
      print(paste("chunk", i, "tokens extracted."))\
      motifs <- extract_motifs(tokens_df, entities = c("*"), pron_as_ap = TRUE, parse_multi_token_entities = T)\
      \
      result[[paste("chunk", i, sep = "_")]] <- motifs\
    \}, error = function(e) \{\
      # log the error message and the chunk number\
      error_log[[paste("chunk", i, sep = "_")]] <- e$message\
      print(e$message)\
    \})\
  \}\
  \
  # combine all motifs data frames from each chunk.\
  # initialize an empty list for each category\
  combined_actions <- list()\
  combined_treatments <- list()\
  combined_characterizations <- list()\
  combined_possessions <- list()\
  combined_agent_treatments <- list()\
  combined_action_patients <- list()\
  \
  # loop thru each category of each chunk in the chunk list and combine using do.call with rbind\
  for (i in seq_along(result)) \{\
    combined_actions[[i]] <- result[[i]]$actions\
    combined_treatments[[i]] <- result[[i]]$treatments\
    combined_characterizations[[i]] <- result[[i]]$characterizations\
    combined_possessions[[i]] <- result[[i]]$possessions\
    combined_agent_treatments[[i]] <- result[[i]]$agent_treatments\
    combined_action_patients[[i]] <- result[[i]]$action_patients\
  \}\
  \
  # combine each category\
  final_actions <- do.call(rbind, combined_actions)\
  final_treatments <- do.call(rbind, combined_treatments)\
  final_characterizations <- do.call(rbind, combined_characterizations)\
  final_possessions <- do.call(rbind, combined_possessions)\
  final_agent_treatments <- do.call(rbind, combined_agent_treatments)\
  final_action_patients <- do.call(rbind, combined_action_patients)\
  \
  # write final list of combined data frames\
  final_combined_list <- list(\
    actions = final_actions,\
    treatments = final_treatments,\
    characterizations = final_characterizations,\
    possessions = final_possessions,\
    agent_treatments = final_agent_treatments,\
    action_patients = final_action_patients\
  )\
  \
  # return combined results.\
  print("motifs for one work extracted")\
  print(paste("Memory usage after extracting one work's motif:", pryr::mem_used()))\
  return(final_combined_list)\
  \
\}\
\
# Time the operation of applying formatting_text function\
time_formatting <- system.time(\{\
  corpus[!is.na(body) & body != '', body := lapply(.SD, formatting_text), .SDcols = "body"]\
\})\
\
# Print the timing information for formatting\
print("Time for formatting:")\
print(time_formatting)\
\
# applying process_books function to create motif\
time_motif_creation <- system.time(\{\
  corpus[, motif := ifelse(!is.na(body) & body != '', lapply(body, process_books), '')]\
\})\
\
\
print(paste("Memory after all motifs are extracted for this corpus:", pryr::mem_used()))\
\
# Define the output filename for JSON\
output_name <- sprintf("corpus_chunks_results/corpus_chunk_%02d_with_motifs.json", task_id)\
\
# Save the data frame or list as JSON\
write_json(corpus, path = output_name)}