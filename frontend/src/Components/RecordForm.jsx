import { useLocation,useNavigate } from 'react-router-dom';
import styled from 'styled-components';
import React, { useState ,useEffect } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faMicrophone } from '@fortawesome/free-solid-svg-icons';
import Modal from './Modal';
import axios from 'axios';
import { useAuthContext } from '../utils/authContext';

const FormContainer = styled.div`
  background: #fff;
  padding: 20px;
  box-shadow: 0 0 10px rgba(0,0,0,0.1);
  border-radius: 10px;
  width: 100%;
`;

const FormGroup = styled.div`
  margin-bottom: 20px;
`;

const Label = styled.label`
  display: block;
  margin-bottom: 5px;
  font-weight: bold;
`;

const Input = styled.input`
  width: 100%;
  padding: 10px;
  margin-bottom: 10px;
  border: 1px solid #ccc;
  border-radius: 5px;
`;

const Select = styled.select`
  width: 100%;
  padding: 10px;
  margin-bottom: 10px;
  border: 1px solid #ccc;
  border-radius: 5px;
`;

const CheckboxGroup = styled.div`
  display: flex;
  flex-wrap: wrap;

  label {
    width: 50%;
    margin-bottom: 10px;
  }
`;

const Table = styled.table`
  width: 100%;
  border-collapse: collapse;
  margin-bottom: 20px;

  th, td {
    border: 1px solid #ccc;
    padding: 10px;
  }

  th {
    background: #f4f4f4;
  }

  td {
    text-align: center;
  }
`;

const AddMedicineButton = styled.button`
  background: #4caf50;
  color: white;
  padding: 10px 20px;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  transition: background 0.3s;

  &:hover {
    background: #45a049;
  }
`;

const Attachments = styled.div`
  display: flex;
  flex-wrap: wrap;
  margin-bottom: 20px;

  img {
    margin-right: 10px;
    margin-bottom: 10px;
    border: 1px solid #ccc;
    border-radius: 5px;
  }
`;

const Upload = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  border: 2px dashed #ccc;
  border-radius: 5px;
  padding: 20px;
  cursor: pointer;

  input {
    display: none;
  }

  p {
    margin: 0;
  }
`;

const SaveButton = styled.button`
  background: #008cba;
  color: white;
  padding: 10px 20px;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  transition: background 0.3s;

  &:hover {
    background: #007bb5;
  }
`;

const MedicalRecordForm = () => {
  const location = useLocation();
  const pid = localStorage.getItem('pid');
  const { authUser } = useAuthContext();
  const navigate = useNavigate();
  const [medicines, setMedicines] = useState([
    { name: 'Paracetamol', price: 1000, dosage: '1 - M/A/E', instruction: 'After meal', quantity: 1, amount: 1000 },
    { name: 'Amoxicillin', price: 2300, dosage: '2 - M/A/E', instruction: 'After meal', quantity: 2, amount: 4600 },
    { name: 'Ibuprofen', price: 5000, dosage: '3 - M/A/E', instruction: 'Before meal', quantity: 3, amount: 15000 },
  ]);
  const [audio,setAudio] = useState(null);
  const [text,setText] = useState(null);
  const [summary,setSummary] = useState([]);

  useEffect(() => {
    const storedAudio = localStorage.getItem('recordedAudio');
    if (storedAudio) {
      setAudio(storedAudio);
    }
  }, []);

  useEffect(() => {
    const fetchSummary = async () => {
      try {
        const response = await axios.get(`http://localhost:3001/api/auth/getSummary/${pid}`);
        setSummary(response.data.data);
      } catch (error) {
        console.error('Error fetching patients:', error);
      }
    };
    fetchSummary();
  }, []);

  if(audio){
    const stt = async () => {
    try{
      const {data} = await axios.get('http://localhost:3001/api/auth/openai/getText');
      localStorage.setItem('output',data.text);
      setText(data.text);
      const res = await axios.post('http://localhost:3001/api/auth/saveReport',{ 
        doctorId: authUser._id,
        patientId: pid,
        summary: data.text,
      },{withCredentials:true});
      console.log(res.data);
    }catch (error) {
      console.log(error);
      }
    }
    stt();
  }

  const [isModalOpen, setModalOpen] = useState(false);

  const handleSaveMedicine = (medicine) => {
    setMedicines([...medicines, medicine]);
    setModalOpen(false);
  };

  const handleClick = () => {
    navigate('/record/');
  }

  const handleSave = () => {
    navigate('/report',{data:text});
  }

  console.log(summary);

  return (
    <FormContainer>
      <form className='flex flex-col justify-center'>
        <FormGroup>
          <Label className='text-lg'>Proffesor</Label>
          <p className='text-3xl'>Prof.{authUser.fullName}</p>
        </FormGroup>
        <div className='bg-green-normal rounded-md flex flex-col justify-center items-center'>
        {audio ? (
            <div>{text}</div>
          ) : (
            <div className='flex flex-col justify-center items-center'>
              <span className='text-white text-lg mt-4 mb-2'>Diagnosis</span>
              <FontAwesomeIcon onClick={handleClick} icon={faMicrophone} className='size-10 pb-2 text-white mt-2 mb-4' />
            </div>
          )}
        </div>
        <Table className='mt-4'>
        </Table>
        <SaveButton onClick={handleSave} className='self-center'>Save</SaveButton>
        <div className="report-container">
            <h1 className='text-3xl'>Lecture Summaries</h1>
            {summary.length > 0 ? (
                <div className="summ-list">
                    {summary.map((summ, index) => (
                        <div key={index} className="summary-item bg-green-normal m-2 text-white rounded-md flex flex-col justify-center items-center">
                            <div>{summ.summary}</div>
                        </div>
                    ))}
                </div>
            ) : (
                <p>No summaries available.</p>
            )}
        </div>
      </form>
      {isModalOpen && <Modal onSave={handleSaveMedicine} onClose={() => setModalOpen(false)} />}
    </FormContainer>
  );
}

export default MedicalRecordForm;