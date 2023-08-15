# TEST Instruccions 

This is the documentation to perform test on this project fuctionalities

## Testing wheather the dataset path contains the dataset 

The following command test the load functionality that works in load module

```
pytest .\test\test_file.py::test_csv_file_existence
```

Remember that after executing the test to delete the dataset from the data folder
in order to not commit the project with a large file
