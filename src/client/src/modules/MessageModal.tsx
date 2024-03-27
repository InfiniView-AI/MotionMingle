import React from 'react';
import { Modal, Box, Typography, Button } from '@mui/material';

const style = {
  position: 'absolute' as const,
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  width: 400,
  bgcolor: 'background.paper',
  boxShadow: 24,
  p: 4,
};

export default function MessageModal(props: {
  isModalOpen: boolean;
  handleClose: () => void;
  handelStopVideo: () => void;
}) {
  const { isModalOpen, handleClose, handelStopVideo } = props;
  return (
    <Modal open={isModalOpen} onClose={handleClose}>
      <Box sx={style}>
        <Typography id="modal-modal-title" variant="h6" component="h2">
          Warning
        </Typography>
        <Typography id="modal-modal-description" sx={{ mt: 2 }}>
          Are you sure you want to stop your video stream? Practitioner in this
          room can not see your live stream after you stop the video.
        </Typography>
        <Button variant="contained" color="primary" onClick={handelStopVideo}>
          Stop Video
        </Button>
        <Button variant="contained" color="error" onClick={handleClose}>
          Cancel
        </Button>
      </Box>
    </Modal>
  );
}
