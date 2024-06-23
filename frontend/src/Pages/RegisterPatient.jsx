/* import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import logo from '../Images/logo.jpeg';
import axios from 'axios';
import { useAuthContext } from '../utils/authContext';


const RegisterPatient = () => {
    const navigate = useNavigate();
    const { setAuthUser } = useAuthContext();
    const [signupError, setSignupError] = useState('');

    const [signUpUserDetails, setSignUpUserDetails] = useState({
		fullName: "",
        address: "",
        gender:"",
        age:"",
        number:"",
	});

    const handleChange = (event) => {
        const { name, value } = event.target;
        setSignUpUserDetails(prevUserDetails => ({
          ...prevUserDetails,
          [name]: value,
        }))
      }
    
    const handleClick = async (event) => {
        event.preventDefault();
        try {
        const { data } = await axios.post('http://localhost:3001/api/auth/register', { ...signUpUserDetails }, {
            withCredentials: true
        });
        alert(data.message);
        navigate('/home');
        } catch (error) {
        setSignupError(error.response.data);
        setTimeout(() => {
            setSignupError('');
        }, 3000);
        }
    }

    return (
      <>
        <div className="flex min-h-full flex-1 flex-col justify-center px-6 py-12 lg:px-8 text-white">
          <div className="sm:mx-auto sm:w-full sm:max-w-sm">
            <img
              className="mx-auto size-56 w-auto"
              src={logo}
              alt="Your Company"
            />
            <h2 className="mt-10 text-center text-2xl font-bold leading-9 tracking-tight text-black">
              Register Patient
            </h2>
          </div>
  
          <div className="mt-10 sm:mx-auto sm:w-full sm:max-w-sm">
            <form className="space-y-6" action="#" method="POST">
              <div>
                <label htmlFor="email" className="block text-sm font-medium leading-6 text-black">
                  Full Name
                </label>
                <div className="mt-2">
                  <input
                    id="fullName"
                    name="fullName"
                    type="text"
                    value={signUpUserDetails.fullName}
                    onChange={handleChange}
                    required
                    className="block w-full rounded-md border-0 py-1.5 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:text-green-200 sm:text-sm sm:leading-6"
                  />
                </div>
              </div>
  
              <div>
                <div className="flex items-center justify-between">
                  <label htmlFor="password" className="block text-sm text-black font-medium leading-6">
                    Address
                  </label>
                </div>
                <div className="mt-2">
                  <input
                    id="adress"
                    name="address"
                    type="text"
                    required
                    onChange={handleChange}
                    value={signUpUserDetails.address}
                    className="block w-full rounded-md border-0 py-1.5 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:text-green-200 sm:text-sm sm:leading-6"
                  />
                </div>
              </div>

              <div>
                <div className="flex items-center justify-between">
                  <label htmlFor="password" className="block text-sm text-black font-medium leading-6">
                    Gender
                  </label>
                </div>
                <div className="mt-2">
                  <input
                    id="gender"
                    name="gender"
                    type="text"
                    required
                    onChange={handleChange}
                    value={signUpUserDetails.gender}
                    className="block w-full rounded-md border-0 py-1.5 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:text-green-200 sm:text-sm sm:leading-6"
                  />
                </div>
              </div>

              <div>
                <div className="flex items-center text-black justify-between">
                  <label htmlFor="password" className="block text-sm font-medium leading-6">
                    Number
                  </label>
                </div>
                <div className="mt-2">
                  <input
                    id="number"
                    name="number"
                    type="text"
                    required
                    onChange={handleChange}
                    value={signUpUserDetails.number}
                    className="block w-full rounded-md border-0 py-1.5 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:text-green-200 sm:text-sm sm:leading-6"
                  />
                </div>
              </div>

              <div>
                <div className="flex items-center text-black justify-between">
                  <label htmlFor="password" className="block text-sm font-medium leading-6">
                    Age
                  </label>
                </div>
                <div className="mt-2">
                  <input
                    id="age"
                    name="age"
                    type="text" 
                    onChange={handleChange}
                    required
                    value={signUpUserDetails.age}
                    className="block w-full rounded-md border-0 py-1.5 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:text-green-200 sm:text-sm sm:leading-6"
                  />
                </div>
              </div>
  
              <div>
                <button
                  type="submit"
                  className="flex w-full justify-center rounded-md bg-green-normal px-3 py-1.5 text-sm font-semibold leading-6 text-white shadow-sm hover:bg-green-200 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-green-200"
                  onClick={handleClick}
                >
                  Register
                </button>
              </div>
              <span className='error'>{signupError}</span>
            </form>
          </div>
        </div>
      </>
    )
  }
  
export default RegisterPatient;
*/