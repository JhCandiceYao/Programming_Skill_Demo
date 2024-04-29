# My own tutorial for rclone data transfer workflow between HPC and remote(google drive)

#computational_text_analysis #python 

NYU HPC tutorial: (this is generally, if not entirely, useless)
https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/hpc-storage/data-management/data-transfers/transferring-cloud-storage-data-with-rclone
# 1.  Setting up 
-  SSH into HPC first 
- Load rclone module
```
#!/bin/bash
$ module load rclone/1.60.1
# as to 2023.8.30 this is the latest version we could use on NYU HPC cluster - check rclone documentary
```
alternatively, on macbook terminal, you might need an extra step 
```
#!/bin/bash
$ module spider rclone # spider first
$ module load rclone/1.60.1
```

- rclone configuration
	- Theoretically this only needs to be done once, but authentification could expire
```
$ rclone config
```

Follow steps. This part should be easy.
## *Pay extra attention to this part!!!:
- This is quite tricky because when I missed the instruction on this specific question and thought that it should be defaulted to 'yes' as other questions, I did not get to see the config token prompts and was tripping somewhere else.
- Example prompt snippets:
- Use auto config? * Say Y if not sure * Say N if you are working on a remote or headless machiney) Yes (default)n) No y/n> n
	Please go to the following link: https://accounts.google.com/o/oauth2/auth?access_type=offline&client_id=  CUT AND PASTE The URL ABOVE INTO A BROWSER ON YOUR LAPTOP/DESKTOP Log in and authorize rclone for accessEnter verification code> ENTER VERIFICATION CODE HERE(copied from NYU HPC tutorial)
	- Note that while most of the other option can be defaulted, this is the only question that requires us to do a non-default selection. 
- example:
-  this is what the process looks like in Greene shell 
![[Pasted image 20230830171453.png]]
- This is what it will look like on my local machine's terminal, after running what was copied and pasted from the Greene shell in my local terminal, there should be a Google Grive authentification page popping up.
![[Pasted image 20230830171431.png]]
- After obtaining the config token(indicated in the red box section above), copy it back into rclone prompt and finish the config process. 

## *In theory, since the configuration was already done once, the entire workflow of accessing rclone can be as simple as:*

```
$ module load rclone/1.60.1
$ rclone lsd remote1:
```

![[Pasted image 20230830172346.png]]

rclone copy remote1:/UGthesis/CORPUS/download/chunks_n2000 /scratch/jy3440/MOTIFS/corpus_chunks/
# 2. Transferring files between source and dest 
- Basically everything can be built from the syntax below: 
```
# copying an entire directory
$ rclone copy <source> <dest>
# from gdrive to HPC home
$ rclone copy remote1:api_data /home/jy3440/
# from HPC home to gdrive
$ rclone copy /home/jy3440/xyzxyz.csv remote1:api_data
# copying a single file 
$ rclone copy remote1:Locate_Climate_Tweets.ipynb /home/jy3440/
```

**- Note that the syntax of the rclone remote and the local directory is different. Rclone's folder structure is signified by ":"**
(e.g. )
```
rclone lsd remote1:  #	- keep the ":" here, it couldn't be removed, otherwise it won't work!!
```


## Some options/flags for copying:
- Use the [--no-traverse](/docs/#no-traverse) option for controlling whether rclone lists the destination directory or not. Supplying this option when copying a small number of files into a large destination can speed transfers up greatly. For example, if you have many files in /path/to/src but only a few of them change every day, you can copy all the files which have changed recently very efficiently like this:
	- `rclone copy --max-age 24h --no-traverse /path/to/src remote:`
- Use the `-P`/`--progress` flag to view real-time transfer statistics. (*Basically a progress bar, make sure to include this.*)
- Use the `--dry-run` or the `--interactive`/`-i` flag to test transferring without copying anything.
- `--create-empty-src-dirs`   Create empty source dirs on destination after copy
 -  `-h, --help`   
## Difference between sync and copy
- [rclone copy](https://rclone.org/commands/rclone_copy/) - Copy files from source to dest, skipping already copied.
	- since this skip already existing files, it could function exactly the same way as sync 
- [rclone sync](https://rclone.org/commands/rclone_sync/) - Make source and dest identical, modifying destination only.
-> seems like in my current use case(syncing files in google drive from HPC cluster) both could work.

- There is also [rclone bisync](https://rclone.org/commands/rclone_bisync/) - [Bidirectional synchronization](https://rclone.org/bisync/) between two paths.
	- But this seems to be a bit risky for now(before I fully familiarize myself with this system syntax) so i'm just gonna leave it here  

# 3. Miscellaneous 
- [rclone ls](https://rclone.org/commands/rclone_ls/) - List all the objects in the path with size and path.
- [rclone lsd](https://rclone.org/commands/rclone_lsd/) - List all directories/containers/buckets in the path.

- seems like unix command <cd> doesn't work here, so when every I want to operate on a file, I can only directly specify its paths 

