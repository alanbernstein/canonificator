CLI/light GUI tool for custom photo deduplication process, based on the concept of a "canonical" directory

## motivation
I want to deduplicate all of my extraneous photo backups, with two special circumstances:

1. After a data loss event, I used software like photorec and extundelete to recover many files. In most cases, file attributes, folder structure, and/or EXIF tags were NOT recovered. In some cases, the files were corrupted. In many cases, the content of the photo is still as good as the original, which means that the content itself can be used to compare with my canonical directory.
2. Prior to the above event, I generated a 10% scale copy of all photos, in a mirrored directory structure. These images can be used to scan for files that were truly recovered from the void.


## implementation
1. Scan everything and put it in a sqlite database. include fields like md5sum, perceptual hash, dimensions, etc. Use these fields to generate lists of candidate duplicates or recovered files.
2. Use a simple interactive display to show comparisons, with hash-match details, and accept user input to decide whether to delete immediately or save for later review.
