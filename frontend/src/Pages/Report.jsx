import { useEffect, useState } from "react";
import { useLocation } from "react-router-dom"
import axios from 'axios';

const Report = () => {
    const [report,setReport] = useState();
    const [patientDetails,setPatientDetails] = useState();
    const data = localStorage.getItem('output');
    
    useEffect(() => {
        const fetchData = async () => {
            try {
                console.log('data');
                const response = await axios.post('http://localhost:3001/api/auth/openai/getReport', { data }, {
                    withCredentials: true
                  });
                setReport(response.data);
            } catch (error) {
                console.log(error);
            }
            localStorage.removeItem('output'); 
        }
        fetchData();
    },[])

    useEffect(() => {
        const fetchPatients = async () => {
          try {
            const id = localStorage.getItem('pid');
            const response = await axios.get(`http://localhost:3001/api/auth/getPatient/${id}`);
            setPatientDetails(response.data.data);
          } catch (error) {
            console.error('Error fetching patients:', error);
          }
        };
    
        fetchPatients();
      }, []);

    return (
    <div className="report-container bg-white p-4 rounded-md shadow-md">
        {report ? (
        <div className="report-content text-black">
            <h2 className="text-3xl text-black font-bold mb-4">Class Recap</h2>
            <h3>Name: {patientDetails && patientDetails.fullName}</h3>
            <h3>Ph-Number: {patientDetails && patientDetails.number}</h3>
            <h3>Age: {patientDetails && patientDetails.age}</h3>
            <table className="min-w-full bg-white">
            <thead>
                <tr>
                <th className="py-2 px-4 border-b text-black text-2xl">Description</th>
                <th className="py-2 px-4 border-b text-black text-2xl">Content</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                <td className="py-2 px-4 border-b text-black text-lg">Summary</td>
                <td className="py-2 px-4 border-b">{report.summary}</td>
                </tr>
                <tr>
                {/* <td className="py-2 px-4 border-b text-black text-lg">Prescription</td> */}
                {/* <td className="py-2 px-4 border-b">{report.summary2}</td> */}
                </tr>
                <tr>
                <td className="py-2 px-4 border-b text-black text-lg">Hindi Translation</td>
                <td className="py-2 px-4 border-b">{report.hindiTranslatation}</td>
                </tr>
                <tr>
                <td className="py-2 px-4 border-b text-black text-lg">Kannada Translation</td>
                <td className="py-2 px-4 border-b">{report.kannadaTranslation}</td>
                </tr>
            </tbody>
            </table>
        </div>
        ) : (
        <div className="text-center text-gray-500">Loading report...</div>
        )}
    </div>
    );    
    }

export default Report;