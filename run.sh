#!/bin/bash 
has_d_option=false
while getopts :hd opt; do
    case $opt in 
        h) echo "help"; exit ;;
        d) has_d_option=true ;;
        :) echo "Missing argument for option -$OPTARG"; exit 1;;
       \?) echo "Unknown option -$OPTARG"; exit 1;;
    esac
done
shift $(( OPTIND - 1 ))

if $has_d_option; then
    # echo "Debugging..."
    python main.py $1
else
    python main.py $1 $2
fi