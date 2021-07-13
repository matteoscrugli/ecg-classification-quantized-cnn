#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>

void printBytes(unsigned char* buff, int len);



#define DATA_PATH 			"output/dataset/raw_text/"
#define TH_PATH				"output/threshold/default/"
#define ECG_RATE 			360
#define SAMPLES_PERFILE		650000
#define FILES				48
#define DSP_BUFFERDIM		SAMPLES_PERFILE



u_int32_t			threshold;
int					ecg_state = 0;
u_int32_t			ecg_rawSignal_rPeak;
u_int32_t			ecg_heartbeat;



#define 			DSP_ALGORITHM_1



#ifdef ALGORITHM_X
	#define			DSP_DC_FILTER
	#define			DSP_LP_FILTER
	#define			DSP_HP_FILTER
	#define 		DSP_DERIVATIVE
	#define 		DSP_SQUARED
	#define 		DSP_INTEGRAL
#endif

#ifdef DSP_ALGORITHM_1
	#define			DSP_DC_FILTER
	#define			DSP_LP_FILTER
//	#define			DSP_HP_FILTER
	#define 		DSP_DERIVATIVE
	#define 		DSP_SQUARED
//	#define 		DSP_INTEGRAL

#define				LP_FILTER_GAIN
#define				HP_FILTER_GAIN
#define				DERIVATIVE_GAIN 		/ ((float) (8 * 36))

#define				SIGNAL_RISE_THRESHOLD	200000

const unsigned int	signal_peak_delay = 	ECG_RATE / ((float) 4);
#endif



int32_t 			ecg_rawSignal			[DSP_BUFFERDIM] = {0};

#ifdef DSP_DC_FILTER
int32_t 			ecg_dcfilter			[DSP_BUFFERDIM] = {0};
int					ecg_dcfilter_delay = 	0;
#define				DC_FILTER_WINDOW		1
#define				DC_FILTER_DELAY 		0
#endif

#ifdef DSP_LP_FILTER
int32_t 			ecg_lowpass				[DSP_BUFFERDIM] = {0};
int					ecg_lowpass_delay = 	0;
#define				LP_FILTER_WINDOW		12
#define				LP_FILTER_DELAY 		5
#endif

#ifdef DSP_HP_FILTER
int32_t 			ecg_highpass			[DSP_BUFFERDIM] = {0};
int					ecg_highpass_delay = 	0;
#define				HP_FILTER_WINDOW 		32
#define				HP_FILTER_DELAY 		16
#endif

#ifdef DSP_DERIVATIVE
int32_t 			ecg_derivative 			[DSP_BUFFERDIM] = {0};
int					ecg_derivative_delay = 	0;
#define				DERIVATIVE_WINDOW 		5
#define				DERIVATIVE_DELAY 		2
#endif

#ifdef DSP_SQUARED
int32_t 			ecg_squared				[DSP_BUFFERDIM] = {0};
int					ecg_squared_delay = 	0;
#define				SQUARED_WINDOW 			1
#define				SQUARED_DELAY 			0
#endif

#ifdef DSP_INTEGRAL
int32_t 			ecg_integral			[DSP_BUFFERDIM] = {0};
int					ecg_integral_index;
int					ecg_integral_delay = 	0;
#define				INTEGRAL_WINDOW			20
#define				INTEGRAL_DELAY 			10
#define				INTEGRAL_GAIN 			/ ((float) INTEGRAL_WINDOW)
#endif

int32_t				*ecg_signal;
u_int32_t			ecg_signal_index = 		0;
int					ecg_signal_delay = 		0;

#define		 		PRE_BUFFERING 			CNN_HALFWINDOWDIM

int 				dsp_start = 			0;



int main()
{
	struct dirent 			*de;
	FILE 					*fp;
	FILE 					*ft;
	FILE 					*fd;
	FILE 					*ff;
	FILE 					*fp_peak;
	char 					data_delay[512];
	char 					data_fileloc[512];
	char 					data_fileloc_peak[512];
	char 					data_fileloc_th[512];
	char 					data_filtered[512];
	size_t 					byte_read;
	uint					file_cnt;
	uint					sample_cnt;

	char 					*data_subfilename = 	"intdata";
	char 					data_chunk[128];
	int16_t 				data_sample;



#ifdef DSP_DC_FILTER
	ecg_signal_delay +=		DC_FILTER_DELAY;
	ecg_dcfilter_delay =	ecg_signal_delay;
#endif
#ifdef DSP_LP_FILTER
	ecg_signal_delay +=		LP_FILTER_DELAY;
	ecg_lowpass_delay =		ecg_signal_delay;
#endif
#ifdef DSP_HP_FILTER
	ecg_signal_delay +=		HP_FILTER_DELAY;
	ecg_highpass_delay =	ecg_signal_delay;
#endif
#ifdef DSP_DERIVATIVE
	ecg_signal_delay +=		DERIVATIVE_DELAY;
	ecg_derivative_delay =	ecg_signal_delay;
#endif
#ifdef DSP_SQUARED
	ecg_signal_delay +=		SQUARED_DELAY;
	ecg_squared_delay =		ecg_signal_delay;
#endif
#ifdef DSP_INTEGRAL
	ecg_signal_delay +=		INTEGRAL_DELAY;
	ecg_integral_delay =	ecg_signal_delay;
#endif



	DIR *dr = opendir(DATA_PATH);

	if (dr == NULL)  // opendir returns NULL if couldn't open directory
	{
		printf("Could not open current directory" );
		return 0;
	}

	file_cnt = 0;
	sample_cnt = 0;

	sprintf(data_delay, DATA_PATH"filter_delay.txt");
	fd = fopen(data_delay, "w");
	fprintf(fd,"%d\n", ecg_signal_delay);
	fclose(fd);

	while ((de = readdir(dr)) != NULL)
	{
		if(de->d_name[0] != '.' && strstr(de->d_name, data_subfilename) != NULL && file_cnt < FILES)
		{
			sprintf(data_fileloc, DATA_PATH"%s", de->d_name);

			fp = fopen(data_fileloc, "r");

			sprintf(data_fileloc_th, TH_PATH"%c%c%c_th.txt", de->d_name[0], de->d_name[1], de->d_name[2]);
			ft = fopen(data_fileloc_th, "r");
			fscanf(ft, "%d", &threshold);
			fclose(ft);

			sprintf(data_fileloc_peak, DATA_PATH"%c%c%c_peakpos.txt", de->d_name[0], de->d_name[1], de->d_name[2]);
			fp_peak = fopen(data_fileloc_peak, "w");

			sprintf(data_filtered, DATA_PATH"%c%c%c_filtered.txt", de->d_name[0], de->d_name[1], de->d_name[2]);
			ff = fopen(data_filtered, "w");

			while(fgets(data_chunk, sizeof(data_chunk), fp) != NULL && sample_cnt < SAMPLES_PERFILE)
			{
				sample_cnt++;
				sscanf(data_chunk,"%hd\n", &data_sample);



				ecg_rawSignal[ecg_signal_index % DSP_BUFFERDIM] = data_sample;

				ecg_signal = ecg_rawSignal;



#ifdef DSP_DC_FILTER
				if (ecg_signal_index >= dsp_start + 1)
				{
					ecg_dcfilter[ecg_signal_index % DSP_BUFFERDIM] = 	ecg_signal[ecg_signal_index % DSP_BUFFERDIM]
																		- ecg_signal[(ecg_signal_index - 1) % DSP_BUFFERDIM]
																		+ 0.995 * ecg_dcfilter[(ecg_signal_index - 1) % DSP_BUFFERDIM];
				}

				ecg_signal = ecg_dcfilter;
#endif



#ifdef DSP_LP_FILTER
				if (ecg_signal_index >= dsp_start + 12)
				{
					ecg_lowpass[ecg_signal_index % DSP_BUFFERDIM] = 	ecg_signal[ecg_signal_index % DSP_BUFFERDIM] LP_FILTER_GAIN
																		- 2 * ecg_signal[(ecg_signal_index - 6) % DSP_BUFFERDIM] LP_FILTER_GAIN
																		+ ecg_signal[(ecg_signal_index - 12) % DSP_BUFFERDIM] LP_FILTER_GAIN
																		+ 2 * ecg_lowpass[(ecg_signal_index - 1) % DSP_BUFFERDIM]
																		- ecg_lowpass[(ecg_signal_index - 2) % DSP_BUFFERDIM];
				}
				else if (ecg_signal_index >= dsp_start + 6)
				{
					ecg_lowpass[ecg_signal_index % DSP_BUFFERDIM] = 	ecg_signal[ecg_signal_index % DSP_BUFFERDIM] LP_FILTER_GAIN
																		- 2 * ecg_signal[(ecg_signal_index - 6) % DSP_BUFFERDIM] LP_FILTER_GAIN
																		+ 2 * ecg_lowpass[(ecg_signal_index - 1) % DSP_BUFFERDIM]
																		- ecg_lowpass[(ecg_signal_index - 2) % DSP_BUFFERDIM];
				}
				else if (ecg_signal_index >= dsp_start + 2)
				{
					ecg_lowpass[ecg_signal_index % DSP_BUFFERDIM] = 	ecg_signal[ecg_signal_index % DSP_BUFFERDIM] LP_FILTER_GAIN
																		+ 2 * ecg_lowpass[(ecg_signal_index - 1) % DSP_BUFFERDIM]
																		- ecg_lowpass[(ecg_signal_index - 2) % DSP_BUFFERDIM];
				}
				else if (ecg_signal_index >= dsp_start + 1)
				{
					ecg_lowpass[ecg_signal_index % DSP_BUFFERDIM] = 	ecg_signal[ecg_signal_index % DSP_BUFFERDIM] LP_FILTER_GAIN
																		+ 2 * ecg_lowpass[(ecg_signal_index - 1) % DSP_BUFFERDIM];
				}
				else if (ecg_signal_index >= dsp_start)
				{
					ecg_lowpass[ecg_signal_index % DSP_BUFFERDIM] = 	ecg_signal[ecg_signal_index % DSP_BUFFERDIM] LP_FILTER_GAIN;
				}

				ecg_signal = ecg_lowpass;
#endif



#ifdef DSP_HP_FILTER
				if (ecg_signal_index >= dsp_start + 32)
				{
					ecg_highpass[ecg_signal_index % DSP_BUFFERDIM] = 	- ecg_signal[ecg_signal_index % DSP_BUFFERDIM] HP_FILTER_GAIN
																		+ 32 * ecg_signal[(ecg_signal_index - 16) % DSP_BUFFERDIM] HP_FILTER_GAIN
																		+ ecg_signal[(ecg_signal_index - 32) % DSP_BUFFERDIM] HP_FILTER_GAIN
																		- ecg_highpass[(ecg_signal_index - 1) % DSP_BUFFERDIM];
				}
				else if (ecg_signal_index >= dsp_start + 16)
				{
					ecg_highpass[ecg_signal_index % DSP_BUFFERDIM] = 	- ecg_signal[ecg_signal_index % DSP_BUFFERDIM] HP_FILTER_GAIN
																		+ 32 * ecg_signal[(ecg_signal_index - 16) % DSP_BUFFERDIM] HP_FILTER_GAIN
																		- ecg_highpass[(ecg_signal_index - 1) % DSP_BUFFERDIM];
				}
				else if (ecg_signal_index >= dsp_start + 1)
				{
					ecg_highpass[ecg_signal_index % DSP_BUFFERDIM] = 	- ecg_signal[ecg_signal_index % DSP_BUFFERDIM] HP_FILTER_GAIN
																		- ecg_highpass[(ecg_signal_index - 1) % DSP_BUFFERDIM];
				}
				else if (ecg_signal_index >= dsp_start)
				{
					ecg_highpass[ecg_signal_index % DSP_BUFFERDIM] = 	- ecg_signal[ecg_signal_index % DSP_BUFFERDIM] HP_FILTER_GAIN;
				}

				ecg_signal = ecg_highpass;
#endif

#ifdef DSP_DERIVATIVE
				if (ecg_signal_index >= dsp_start + 4)
				{
					ecg_derivative[ecg_signal_index % DSP_BUFFERDIM] = 	- ecg_signal[(ecg_signal_index - 4) % DSP_BUFFERDIM] DERIVATIVE_GAIN
																		- 2 * ecg_signal[(ecg_signal_index - 3) % DSP_BUFFERDIM] DERIVATIVE_GAIN
																		+ 2 * ecg_signal[(ecg_signal_index - 1) % DSP_BUFFERDIM] DERIVATIVE_GAIN
																		+ ecg_signal[ecg_signal_index % DSP_BUFFERDIM] DERIVATIVE_GAIN;
				}
				else if (ecg_signal_index >= dsp_start + 3)
				{
					ecg_derivative[ecg_signal_index % DSP_BUFFERDIM] = 	- 2 * ecg_signal[(ecg_signal_index - 3) % DSP_BUFFERDIM] DERIVATIVE_GAIN
																		+ 2 * ecg_signal[(ecg_signal_index - 1) % DSP_BUFFERDIM] DERIVATIVE_GAIN
																		+ ecg_signal[ecg_signal_index % DSP_BUFFERDIM] DERIVATIVE_GAIN;
				}
				else if (ecg_signal_index >= dsp_start + 1)
				{
					ecg_derivative[ecg_signal_index % DSP_BUFFERDIM] = 	2 * ecg_signal[(ecg_signal_index - 1) % DSP_BUFFERDIM] DERIVATIVE_GAIN
																		+ ecg_signal[ecg_signal_index % DSP_BUFFERDIM] DERIVATIVE_GAIN;
				}
				else if (ecg_signal_index >= dsp_start)
				{
					ecg_derivative[ecg_signal_index % DSP_BUFFERDIM] = 	ecg_signal[ecg_signal_index % DSP_BUFFERDIM] DERIVATIVE_GAIN;
				}

				ecg_signal = ecg_derivative;
#endif

#ifdef DSP_SQUARED
				if (ecg_signal_index >= dsp_start)
				{
					ecg_squared[ecg_signal_index % DSP_BUFFERDIM] = 	ecg_signal[ecg_signal_index % DSP_BUFFERDIM] * ecg_signal[ecg_signal_index % DSP_BUFFERDIM];
				}

				ecg_signal = ecg_squared;
#endif

#ifdef DSP_INTEGRAL
				if (ecg_signal_index >= dsp_start)
				{
					ecg_integral[ecg_signal_index % DSP_BUFFERDIM] = 0;
					for (ecg_integral_index = 0; ecg_integral_index < INTEGRAL_WINDOW; ecg_integral_index++)
					{
						if (ecg_signal_index >= dsp_start + ecg_integral_index)
						{
							ecg_integral[ecg_signal_index % DSP_BUFFERDIM] += ecg_signal[(ecg_signal_index - ecg_integral_index];
						}
						else break;
					}
					ecg_integral[ecg_signal_index % DSP_BUFFERDIM] /= ecg_integral_index;
				}

				ecg_signal = ecg_integral;
#endif

				fprintf(ff,"%d\n", ecg_signal[ecg_signal_index % DSP_BUFFERDIM]);



#ifdef DSP_ALGORITHM_1
				switch (ecg_state & 0b11)
				{
					case (0):
						if(ecg_signal[ecg_signal_index % DSP_BUFFERDIM] >= threshold /*SIGNAL_RISE_THRESHOLD*/) ecg_state++;
						// printf("%d\r\n", ecg_signal[ecg_signal_index]);
					break;

					case (1):
						if(ecg_signal[ecg_signal_index % DSP_BUFFERDIM] < ecg_signal[(ecg_signal_index - 1) % DSP_BUFFERDIM]) ecg_state++;
					break;

					case (2):
						if(ecg_signal[ecg_signal_index % DSP_BUFFERDIM] > ecg_signal[(ecg_signal_index - 1) % DSP_BUFFERDIM])
						{
							ecg_heartbeat = ECG_RATE / ((float) (((ecg_signal_index - 1 - ecg_signal_delay) - ecg_rawSignal_rPeak)));
							ecg_rawSignal_rPeak = (ecg_signal_index - 1 - ecg_signal_delay);
							// printf("%d, ", ecg_rawSignal_rPeak);
							fprintf(fp_peak,"%d\n", ecg_rawSignal_rPeak);

							ecg_state++;
						}
					break;

					case (3):
						if((ecg_signal_index - signal_peak_delay) == ecg_rawSignal_rPeak) ecg_state++;
					break;

					default:
						while(1);
					break;
				}
#endif
			ecg_signal_index++;
			}
			fclose(fp);
			fclose(fp_peak);
			ecg_signal_index = 0;
			ecg_state = 0;
			sample_cnt = 0;
			file_cnt++;
		}
	}

	closedir(dr);



	return 0;
}



void printBytes(unsigned char* buff, int len)
{
    int i;
    for (i = 0; i < len; i++)
    {
        if(i%8 == 0){
            printf("\n");
        }
        printf("0x%02x",buff[i]);
        printf("\t");

    }
}
