#pragma once

#include "integer_sequence.h"

#include <string>

namespace mph
{
	// Wraps a string literal useful for simply storing large help-text strings
	// NOTE: for now can only be used at runtime through implicit conversion
	template <std::size_t N>
	class literal_string
	{
	private:
		const char (&_str)[N+1];

	public:
		constexpr literal_string(const char(&lit_str)[N+1])
			: _str{lit_str}
		{}
	public:
		constexpr std::size_t size() const { return N; }
		constexpr char operator[](int i) const { return _str[i]; }

		constexpr const char* c_str() const { return _str; }

		operator std::string() const { return std::string{ _str }; }
	};

	template <std::size_t N_INC>
	inline constexpr literal_string<N_INC-1> literal(const char(&lit)[N_INC])
	{
		return literal_string<N_INC-1>(lit);
	}


	// Compile-time constant string (note size is very limited in msvc2015)
	template <std::size_t N>
	class const_string
	{
	private:
		const char _str[N+1];

	private:
		template <std::size_t... Is>
		constexpr const_string(mph::index_sequence<Is...>, const char(&lit_str)[N+1])
			: _str{lit_str[Is]..., '\0'}
		{}

		template <std::size_t M, std::size_t... Isa, std::size_t... Isb>
		constexpr const_string(mph::index_sequence<Isa...>, mph::index_sequence<Isb...>,
			const const_string<M>& strA, const const_string<N-M>& strB)
			: _str{strA[Isa]..., strB[Isb]..., '\0'}
		{}

	public:
		constexpr const_string(const char(&lit_str)[N+1])
			: const_string<N>{mph::make_index_sequence<N>(), lit_str}
		{}

		template <std::size_t M>
		constexpr const_string(const const_string<M>& strA, const const_string<N-M>& strB)
			: const_string(mph::make_index_sequence<M>(), mph::make_index_sequence<N-M>(), strA, strB)
		{}

	public:
		constexpr std::size_t size() const {return N;}
		constexpr char operator[](int i) const {return _str[i];}

		constexpr const char* c_str() const {return _str;}

		operator std::string() const {return std::string{_str};}
	};


	template <std::size_t N_INC>
	inline constexpr const_string<N_INC-1> make_const_str(const char(&lit)[N_INC])
	{
		return const_string<N_INC-1>(lit);
	}

	template <std::size_t N>
	inline constexpr const const_string<N>& make_const_str(const const_string<N>& str)
	{
		return str;
	}



	template <std::size_t N, std::size_t M>
	inline constexpr const_string<N+M> operator+(const const_string<N>& strA, const const_string<M>& strB)
	{
		return const_string<N+M>(strA,strB);
	}

	template <std::size_t N_INC, std::size_t M>
	inline constexpr const_string<N_INC+M-1> operator+(const char (&strA)[N_INC], const const_string<M>& strB)
	{
		return const_string<N_INC-1>(strA) + strB;
	}

	template <std::size_t N, std::size_t M_INC>
	inline constexpr const_string<N+M_INC-1> operator+(const const_string<N>& strA, const char(&strB)[M_INC])
	{
		return strA + const_string<M_INC-1>(strB);
	}

	namespace internal
	{
		template <typename T>
		struct conststr_size{};

		template <std::size_t N_INC>
		struct conststr_size<char [N_INC]>
		{
			static constexpr const std::size_t value = N_INC-1;
		};

		template <std::size_t N>
		struct conststr_size<const_string<N>>
		{
			static constexpr const std::size_t value = N;
		};


		template <typename... Strs>
		struct strcat_size {};

		template <typename H, typename... Ts>
		struct strcat_size<H,Ts...>
		{
			using R = typename std::remove_cv<typename std::remove_reference<H>::type>::type;
			static constexpr const std::size_t value = conststr_size<R>::value + strcat_size<Ts...>::value;
		};

		template <>
		struct strcat_size<>
		{
			static constexpr const std::size_t value = 0;
		};

		constexpr const const_string<0> const_strcat_impl()
		{
			return make_const_str("");
		}

		template <typename StrType>
		constexpr const StrType& const_strcat_impl(StrType&& str)
		{
			return str;
		}

		template <typename Head, typename... Tail>
		constexpr auto const_strcat_impl(Head&& h, Tail&&... tail)
			-> const_string<strcat_size<Head, Tail...>::value>
		{
			return std::forward<Head>(h) + const_strcat_impl(std::forward<Tail>(tail)...);
		}


		template <typename Str>
		const std::string& strcat_all_impl(Str&& str)
		{
			return str;
		}

		template <typename Head, typename... Tail>
		std::string strcat_all_impl(Head&& h, Tail&&... tail)
		{
			return std::forward<Head>(h) + strcat_all_impl(std::forward<Tail>(tail)...);
		}
	};

	template <typename... Strs>
	constexpr auto const_strcat(Strs&&... strs)
		-> decltype(internal::const_strcat_impl(std::declval<Strs>()...))
	{
		return internal::const_strcat_impl(std::forward<Strs>(strs)...);
	}

	template <typename... Strs>
	std::string strcat_all(Strs&&... strs)
	{
		return internal:: strcat_all_impl(std::forward<Strs>(strs)...);
	}
};

