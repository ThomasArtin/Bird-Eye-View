#include "calc.h"
//double brake(void);
void calc(double Range_Front,double Elevation_Front,double Azimuth_Front,double ID_front,double Velocity_Front,double Heading_Front \
		 ,double Range_Back,double Elevation_Back,double Azimuth_Back,double ID_Back,double Velocity_Back,double Heading_Back \
		 ,double Range_Left,double Elevation_Left,double Azimuth_Left,double ID_Left,double Velocity_Left,double Heading_Left \
		 ,double Range_Right,double Elevation_Right,double Azimuth_Right,double ID_Right,double Velocity_Right,double Heading_Right \
		 ,double * Velocity,double * Brake,double * Steering )
{
	
	* Velocity=5;
	if(Range_Front<2)
	{
		* Velocity=0;
		* Brake= 150;
		* Steering = 500;
		while (Range_Front <2)
		{
			* Brake= 0;
			* Velocity = 5;
		}
		//* Brake= 150;

		
	}
	
}
			 
/*void calc(double a,double b,double c,double * A,double * B,double * C )
{
	*A=1*a;
	*B=2*b;
	*C=3*c;
	
	if ((*A>100)||(*B>100)||(*C>100))
	{
		*A=ReturnNegativeOne();
		*B=ReturnNegativeOne();
		*C=ReturnNegativeOne();
	}
	
}*/

/*double brake(void)
{
	return 150;
}*/