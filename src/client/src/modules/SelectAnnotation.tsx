import React from 'react';
import {
    Select,
    MenuItem,
    FormControl,
    InputLabel,
    SelectChangeEvent,
  } from '@mui/material';

function SelectAnnotation(props: {
    selectedAnnotation: string; 
    selectionHandler: (event: SelectChangeEvent) => void;
}) {
    const { selectedAnnotation, selectionHandler } = props;

    return (
        <FormControl fullWidth>
        <InputLabel id="demo-simple-select-label">Annotation</InputLabel>
        <Select
          labelId="demo-simple-select-label"
          id="demo-simple-select"
          value={selectedAnnotation}
          label="Selected Annotation"
          onChange={selectionHandler}
        >
          <MenuItem value="">None</MenuItem>
          <MenuItem value="skeleton">Skeleton</MenuItem>
          <MenuItem value="edges">Edges</MenuItem>
          <MenuItem value="cartoon">Cartoon</MenuItem>
          <MenuItem value="segmentation">Segmentation</MenuItem>
        </Select>
      </FormControl>

    )
}

export default SelectAnnotation;