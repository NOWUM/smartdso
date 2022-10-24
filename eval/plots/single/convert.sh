if ! [ -z "$1" ]
then
     echo "$1"
     inkscape $1 --export-area-drawing --batch-process --export-type=pdf --export-filename="${1%.*}.pdf"
else
for file in *.svg; do 
    if [ -f "$file" ]; then 
        echo "$file" 
        filename=$(basename -- $file)
        inkscape $file --export-area-drawing --batch-process --export-type=pdf --export-filename="${filename%.*}.pdf"
    fi 
done
fi
# inkscape architecture_micro.svg --export-area-drawing --batch-process --export-type=pdf --export-filename="architecture_micro.pdf"
