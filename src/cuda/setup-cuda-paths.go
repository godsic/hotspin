package main

import (
	"os"
	"fmt"
	"io/ioutil"
	"flag"
	"text/template"
	"path/filepath"
	"strings"
)

type Config struct{
	Include	[]string
	Lib []string
}


var dir = flag.String("dir", ".", "root of the cuda wrappers")
var config Config

const VARDELIM = ":"
const CUDA_INC_PATH="CUDA_INC_PATH"
const CUDA_LIB_PATH="CUDA_LIB_PATH"

const CONFIG_TEMP_FILE_NAME="config.go.template"
const CONFIG_FILE_NAME="config.go"

func main(){
	flag.Parse()
	
	CUDAINC := filepath.ToSlash(os.Getenv(CUDA_INC_PATH))
	CUDALIB := filepath.ToSlash(os.Getenv(CUDA_LIB_PATH))
	
	config.Include = strings.Split(CUDAINC, VARDELIM)
	config.Lib = strings.Split(CUDALIB, VARDELIM)
	
	fmt.Println(*dir)
	dirs, _ := ioutil.ReadDir(*dir)
	for i, _ := range(dirs) {
		if dirs[i].IsDir() {
			dirName := dirs[i].Name()
			currDir := *dir + "/" + dirName
			fmt.Println(currDir)
			tempPath := currDir + "/" + CONFIG_TEMP_FILE_NAME
			outPath := currDir + "/" + CONFIG_FILE_NAME
			
			t,error := template.ParseFiles(tempPath)
			if error != nil {
				fmt.Println("The folder has no template files")
				continue
			}
			file, err := os.OpenFile(outPath, os.O_TRUNC | os.O_CREATE | os.O_WRONLY, 0666)
			if err != nil {
				panic(err)
			}
			t.Execute(file, config)
			file.Close()
		}
	}
}
