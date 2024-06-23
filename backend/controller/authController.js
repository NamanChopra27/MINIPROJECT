const { StatusCodes } = require('http-status-codes');
const { BadRequestError, NotFoundError } = require('../errors/errors');
const DoctorSchema = require('../models/doctorModel');
const PatientSchema = require('../models/patientModel');
const fs = require('fs');
const Openai = require('openai');
const ReportSchema = require('../models/reportModel');
const {detectLanguage,translateText} = require('./translation');
require('dotenv').config();


// Read stop words from the file
const stopWordsPath = './data/english.txt';
const stopWords = fs.readFileSync(stopWordsPath, 'utf-8')
    .split('\n')
    .map(word => word.trim());

const removeUnwantedWords = (text) => {
    return text.split(' ')
        .filter(word => !stopWords.includes(word.toLowerCase()))
        .join(' ');
}

const login = async (req, res) => {
    const { fullName, password } = req.body;
    if (!fullName || !password)
        throw new BadRequestError('Please Fill In All The Fields');
    const user = await DoctorSchema.findOne({ fullName });
    if (!user)
        throw new NotFoundError('User Does Not Exist');
    const isMatch = await user.comparePasswords(password);
    if (isMatch) {
        const token = user.createJWT();
        res.status(StatusCodes.OK).json({
            data:{
                _id:user._id,
                fullName:user.fullName,
                age:user.age,
                gender:user.gender,
            },
            token
        });
    }
    else
        throw new BadRequestError('Invalid Credentials');
}

const signup = async (req, res) => {
    const { fullName, password, rePassword, gender ,age} = req.body;
    if (!fullName || !password || !rePassword || !gender)
        throw new BadRequestError('Please Fill In All The Fields');
    if (password !== rePassword)
        throw new BadRequestError('Passwords Do Not Match');
    const userDetails = {
        fullName,
        password,
        gender,
        age,
    };
    const findUser = await DoctorSchema.findOne({ fullName });
    if (findUser)
        throw new BadRequestError('UserName Already Exists');
    const user = await DoctorSchema.create(userDetails);
    const token = user.createJWT();
    res.status(StatusCodes.CREATED).json({
        data:{
            _id:user._id,
            fullName:user.fullName,
            age:user.age,
            gender:user.gender,
        },
        token,
    });
}

const register = async (req, res) => {
    const { fullName, address , gender ,age,number} = req.body;
    if (!fullName || !number || !address || !gender || !age)
        throw new BadRequestError('Please Fill In All The Fields');
    const userDetails = {
        fullName,
        number,
        address,
        gender,
        age,
    };
    const findUser = await PatientSchema.findOne({ fullName });
    if (findUser)
        throw new BadRequestError('UserName Already Exists');
    const user = await PatientSchema.create(userDetails);
    res.status(StatusCodes.CREATED).json({
        status:"200",
        message:"Registered successfully"
    });
}

const allPatients = async (req,res) => {
    const patients = await PatientSchema.find();
    res.status(StatusCodes.OK).json({data:patients});
}

const getPatient = async (req,res) => {
    const { id } = req.params;
    const patient = await PatientSchema.findById(id);
    res.status(StatusCodes.OK).json({data:patient});
}

const getText = async (req,res) => {
    console.log("heyy");
    const openai = new Openai({
        apiKey: process.env.API_KEY,
        dangerouslyAllowBrowser: true,
      });

      try {
        console.log("ssup");
        const text = await openai.audio.transcriptions.create({
        file: fs.createReadStream("C:\\Users\\naman\\OneDrive\\Desktop\\MINIPROJECT\\MediEase\\backend\\audio\\audio.webm"),
        model: "whisper-1",
        response_format: "text",
        })
        console.log("hii");
        fs.unlink("C:\\Users\\naman\\OneDrive\\Desktop\\MINIPROJECT\\MediEase\\backend\\audio\\audio.webm", (err) => {
            if (err) {
              console.error('Error removing file:', err);
              return;
            }
            console.log('File removed successfully');
        });
        res.status(StatusCodes.OK).json({text});
    }
    catch(error){
        console.log("byig");
        res.status(StatusCodes.BAD_REQUEST).json({error:"Failed to extract text"});
    }
}

const hindi = async (data) => {
    const targetLanguage = "hi";
    const detectedLanguage = await detectLanguage(data);
    console.log(`Detected language: ${detectedLanguage}`);

    const translation = await translateText(data, targetLanguage);
    return translation;
}

const kannada = async (data) => {
    const targetLanguage = "kn";

    const detectedLanguage = await detectLanguage(data);
    console.log(`Detected language: ${detectedLanguage}`);

    const translation = await translateText(data, targetLanguage);
    return translation;
}

// const getReport = async (req, res) => {
//     const { data } = req.body;
//     const sum = data + ".For this give a layman explaination of the disease the patient is suffering from , to explain a patient attender";
//     const medicine = data + ".Taking this as reference , if medications are prescribed , give the medication name and frequency of the prescribed medicines seperated each with colon and format it appropriately";

//     try {
//         const openai = new Openai({
//             apiKey: process.env.API_KEY,
//             dangerouslyAllowBrowser: true,
//         });

//         const completion = await openai.chat.completions.create({
//             messages: [{ role: "user", content: sum }],
//             model: "gpt-3.5-turbo",
//         });

//         const completion2 = await openai.chat.completions.create({
//             messages: [{ role: "user", content: medicine}],
//             model: "gpt-3.5-turbo",
//         })

//         const summary = completion.choices[0].message.content;
//         const summary2 = completion2.choices[0].message.content;
//         const hindiTranslatation = await hindi(summary);
//         const kannadaTranslation = await kannada(summary);
//         res.status(StatusCodes.OK).json({ 
//             summary,
//             summary2,
//             hindiTranslatation,
//             kannadaTranslation,
//         });
//     } catch (error) {
//         console.error("Error generating report:", error);
//         res.status(StatusCodes.INTERNAL_SERVER_ERROR).json({ error: "Internal server error" });
//     }
// };

const saveReport = async (req,res) => {
    const { patientId, doctorId , summary } = req.body;
    if (!patientId || !doctorId || !summary)
        throw new BadRequestError('Please Fill In All The Fields');
    const userDetails = {
        patientId,
        doctorId,
        summary
    };
    console.log(userDetails);
    const user = await ReportSchema.create(summary);
    res.status(StatusCodes.CREATED).json({
        status:"200",
        message:"Report created successfully"
    });
}

 const getSummary = async (req,res) => {
     try {
         const { id } = req.params;
        const summaries = await ReportSchema.find({ patientId: id }); 
        res.status(StatusCodes.OK).json({ data: summaries });
} catch (error) {
        res.status(StatusCodes.INTERNAL_SERVER_ERROR).json({ error: 'An error occurred while retrieving summaries.' });
    }
 }

// const chat = async (req,res) => {
//     const { text } = req.body;
//     const input = text + ".Give it in short and pointwise";
//     try {
//         const openai = new Openai({
//             apiKey: process.env.API_KEY,
//             dangerouslyAllowBrowser: true,
//         });

//         const completion = await openai.chat.completions.create({
//             messages: [{ role: "user", content: input }],
//             model: "gpt-3.5-turbo",
//         });

//         const summary = completion.choices[0].message.content;
//         res.status(StatusCodes.OK).json({ 
//             summary
//         });
//     } catch (error) {
//         console.error("Error generating report:", error);
//         res.status(StatusCodes.INTERNAL_SERVER_ERROR).json({ error: "Internal server error" });
//     }
// }

const getLectureSummary = async (req, res) => {
    const { data } = req.body;
    const prompt = data + ".Provide a concise summary of the lecture, avoiding slang and unnecessary details.";

    try {
        // Detect language of the input data
        const detectedLanguage = await detectLanguage(data);

        // Translate the prompt to English if it's not already in English
        let translatedPrompt = prompt;
        if (detectedLanguage !== 'en') {
            translatedPrompt = await translateText(prompt, 'en');
        }

        // Initialize OpenAI client
        const openai = new Openai({
            apiKey: process.env.API_KEY,
            dangerouslyAllowBrowser: true,
        });

        // Generate completion using GPT-3 model
        const completion = await openai.chat.completions.create({
            messages: [{ role: "user", content: translatedPrompt }],
            model: "gpt-3.5-turbo",
        });

        // Extract summary from completion response
        const summary = completion.choices[0].message.content;

        // Translate summary back to original language if necessary
        let translatedSummary = summary;
        if (detectedLanguage !== 'en') {
            translatedSummary = await translateText(summary, detectedLanguage);
        }

         // Remove unwanted words Naman
         const cleanSummary = removeUnwantedWords(summary);

        // Send response with translated summary
        res.status(StatusCodes.OK).json({ summary: translatedSummary });
    } catch (error) {
        console.error("Error generating lecture summary:", error);
        res.status(StatusCodes.INTERNAL_SERVER_ERROR).json({ error: "Internal server error" });
    }
};

// Q&A

const lectureQnA = async (req, res) => {
    const { text } = req.body;
    const prompt = text + ".Provide answers to the following questions based on the lecture content.";

    try {
        const openai = new Openai({
            apiKey: process.env.API_KEY,
            dangerouslyAllowBrowser: true,
        });

        const completion = await openai.chat.completions.create({
            messages: [{ role: "user", content: prompt }],
            model: "gpt-3.5-turbo",
        });

        const summary = completion.choices[0].message.content;
        res.status(StatusCodes.OK).json({ summary });
    } catch (error) {
        console.error("Error generating lecture Q&A:", error);
        res.status(StatusCodes.INTERNAL_SERVER_ERROR).json({ error: "Internal server error" });
    }
};

module.exports = {
    login, 
    signup,
    register,
    allPatients,
    getPatient,
    getText,
    saveReport,
    getLectureSummary,
    lectureQnA,
    hindi,
    kannada,
    
    
}